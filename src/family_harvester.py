"""
Family Harvester: Variation-Aware ASIN Discovery

This module implements intelligent product family discovery for Amazon products.
Instead of naively fetching top N keyword matches, it:

1. Searches for unique product families (using singleVariation)
2. Identifies parent ASINs and their variations
3. Explodes each parent to fetch ALL child variations
4. Filters out junk children (low reviews, 3P-only, etc.)
5. Prioritizes children by quality (BSR, reviews, BB ownership)
6. Fills quota with complete product families

The result: When you search "RXBAR", you get ALL 50 RXBAR flavors,
not 1 RXBAR and 99 competitor bars.

Usage:
    from src.family_harvester import harvest_product_families
    
    families_df = harvest_product_families(
        keyword="RXBAR",
        max_asins=100,
        domain="US"
    )
"""

import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import keepa
from functools import lru_cache


# ============================================================================
# CONFIGURATION
# ============================================================================

# Discovery limits
DEFAULT_SEED_LIMIT = 10        # Initial seed products to find
MAX_CHILDREN_PER_PARENT = 50   # Safety cap per product family
DEFAULT_TOTAL_QUOTA = 100      # Total ASINs to return

# Quality thresholds for filtering junk children
MIN_REVIEW_COUNT = 5           # Minimum reviews to be considered valid
MIN_RATING = 2.5               # Minimum star rating
MAX_BSR = 500000               # Maximum sales rank (filter out dead products)

# API settings
BATCH_SIZE = 20                # Keepa batch size for queries
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0
POLITENESS_DELAY = 0.5


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ProductFamily:
    """Represents a product and its variations."""
    parent_asin: str
    parent_title: str
    parent_brand: str
    parent_bsr: int
    child_asins: List[str] = field(default_factory=list)
    child_count: int = 0
    is_variation_parent: bool = False
    category_id: int = 0
    category_path: str = ""
    has_parent_data: bool = False  # True if parent has real price/BSR data
    
    @property
    def all_asins(self) -> List[str]:
        """
        Return all ASINs with actual data.
        
        IMPORTANT: Parent ASINs (productType=5) typically have N/A for price, BSR, etc.
        Only include the parent if it has real data (is a "hero" ASIN).
        Otherwise, only return the child variations.
        """
        if self.is_variation_parent:
            # Pure variation parent - no data, only return children
            return self.child_asins
        
        if not self.has_parent_data and self.child_asins:
            # Parent has no data but has children - use children only
            return self.child_asins
        
        # Parent has data (hero ASIN) or no children - include parent
        if self.child_asins:
            return [self.parent_asin] + self.child_asins
        return [self.parent_asin]
    
    @property
    def family_size(self) -> int:
        """Total products in this family."""
        return len(self.all_asins)


@dataclass 
class HarvestResult:
    """Result of a family harvest operation."""
    asins: List[str]
    families: List[ProductFamily]
    stats: Dict
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert families to a DataFrame for downstream processing."""
        records = []
        for family in self.families:
            for asin in family.all_asins:
                # is_parent is True only if:
                # 1. This ASIN equals the parent_asin
                # 2. The parent has actual data (has_parent_data=True)
                # If has_parent_data=False, the parent ASIN has no data and shouldn't be marked
                is_parent_with_data = (
                    asin == family.parent_asin and 
                    family.has_parent_data and 
                    not family.is_variation_parent
                )
                records.append({
                    "asin": asin,
                    "parent_asin": family.parent_asin,
                    "brand": family.parent_brand,
                    "family_title": family.parent_title,
                    "category_id": family.category_id,
                    "category_path": family.category_path,
                    "is_parent": is_parent_with_data,
                    "family_size": family.family_size
                })
        return pd.DataFrame(records)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_keepa_api_key() -> Optional[str]:
    """Get Keepa API key from secrets or environment."""
    try:
        return st.secrets.get("KEEPA_API_KEY")
    except:
        import os
        return os.getenv("KEEPA_API_KEY")


def get_domain_id(domain: str) -> int:
    """Convert domain string to Keepa domain ID."""
    domain_map = {
        "US": 1, "GB": 2, "DE": 3, "FR": 4, "JP": 5, 
        "CA": 6, "IT": 8, "ES": 9, "IN": 10, "MX": 11, "BR": 12
    }
    return domain_map.get(domain.upper(), 1)


def _keepa_product_finder_query(
    query_json: Dict,
    api_key: str,
    domain: str = "US"
) -> List[str]:
    """
    Execute a Keepa Product Finder query and return matching ASINs.
    
    Args:
        query_json: The query parameters
        api_key: Keepa API key
        domain: Amazon marketplace
        
    Returns:
        List of matching ASINs
    """
    domain_id = get_domain_id(domain)
    url = f"https://api.keepa.com/query?key={api_key}&domain={domain_id}"
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=query_json, timeout=30)
            
            if response.status_code != 200:
                st.warning(f"Keepa API error: {response.status_code}")
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return []
            
            result = response.json()
            return result.get("asinList", [])
            
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                st.error(f"Product Finder query failed: {e}")
                return []
    
    return []


def _fetch_product_details(
    asins: List[str],
    api_key: str,
    domain: str = "US"
) -> List[Dict]:
    """
    Fetch full product details for a list of ASINs.
    
    Args:
        asins: List of ASINs to fetch
        api_key: Keepa API key
        domain: Amazon marketplace
        
    Returns:
        List of product dictionaries
    """
    if not asins:
        return []
    
    api = keepa.Keepa(api_key)
    all_products = []
    
    for i in range(0, len(asins), BATCH_SIZE):
        batch = asins[i:i + BATCH_SIZE]
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # Fetch with stats and rating for filtering
                products = api.query(batch, domain=domain, stats=30, rating=True)
                all_products.extend(products)
                break
            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    st.warning(f"Failed to fetch batch {i//BATCH_SIZE + 1}: {e}")
        
        if i + BATCH_SIZE < len(asins):
            time.sleep(POLITENESS_DELAY)
    
    return all_products


def _extract_variation_asins(product: Dict) -> List[str]:
    """
    Extract child variation ASINs from a parent product.
    
    Keepa stores variations in 'variationCSV' field as comma-separated ASINs.
    Format: "ASIN1,attr1,val1,ASIN2,attr2,val2,..."
    
    Args:
        product: Keepa product dictionary
        
    Returns:
        List of child ASINs
    """
    variation_csv = product.get("variationCSV")
    if not variation_csv:
        return []
    
    # Parse the CSV - every 3rd value starting at 0 is an ASIN
    parts = variation_csv.split(",")
    child_asins = []
    
    # variationCSV format varies - could be just ASINs or ASIN,attr,val triplets
    # Check if first element looks like an ASIN (10 chars, alphanumeric)
    for i, part in enumerate(parts):
        part = part.strip()
        # ASINs are 10 characters, alphanumeric, start with B0 or digit
        if len(part) == 10 and part.isalnum():
            if part.startswith("B0") or part[0].isdigit():
                child_asins.append(part)
    
    return list(set(child_asins))  # Deduplicate


def _is_variation_parent(product: Dict) -> bool:
    """
    Check if a product is a variation parent (productType=5).
    
    Parent ASINs have NO price, NO BSR, NO usable data.
    They are container listings that group child variations.
    
    Args:
        product: Keepa product dictionary
        
    Returns:
        True if this is a parent ASIN that should be excluded
    """
    return product.get("productType", 0) == 5


def _has_usable_data(product: Dict) -> bool:
    """
    Check if a product has real price or BSR data.
    
    Parent ASINs and some dead listings have no data.
    We should only include products that have actual metrics.
    
    Args:
        product: Keepa product dictionary
        
    Returns:
        True if product has price or BSR data
    """
    csv = product.get("csv", [])
    
    # Check BuyBox price (index 18)
    has_bb_price = bool(
        csv and len(csv) > 18 and csv[18] and 
        len(csv[18]) > 0 and csv[18][-1] > 0
    )
    
    # Check BSR (index 3)
    has_bsr = bool(
        csv and len(csv) > 3 and csv[3] and 
        len(csv[3]) > 0 and csv[3][-1] > 0 and csv[3][-1] != -1
    )
    
    # Check Amazon price (index 0)
    has_amazon = bool(
        csv and len(csv) > 0 and csv[0] and 
        len(csv[0]) > 0 and csv[0][-1] > 0
    )
    
    # Check New price (index 1)
    has_new = bool(
        csv and len(csv) > 1 and csv[1] and 
        len(csv[1]) > 0 and csv[1][-1] > 0
    )
    
    return has_bb_price or has_bsr or has_amazon or has_new


def _calculate_quality_score(product: Dict) -> float:
    """
    Calculate a quality score for prioritizing products.
    
    Higher score = better quality (should be included first)
    
    Factors:
    - Reviews: More reviews = more confidence
    - Rating: Higher rating = better product
    - BSR: Lower BSR = more sales
    - Buy Box: Amazon/FBA preferred
    
    Args:
        product: Keepa product dictionary
        
    Returns:
        Quality score (0-100)
    """
    score = 0.0
    
    # Review count (0-30 points)
    review_count = product.get("stats", {}).get("current", [None] * 20)[17] or 0
    if review_count > 1000:
        score += 30
    elif review_count > 100:
        score += 20
    elif review_count > 10:
        score += 10
    elif review_count > 0:
        score += 5
    
    # Rating (0-20 points)
    rating = product.get("stats", {}).get("current", [None] * 20)[16] or 0
    if rating > 0:
        rating_stars = rating / 10  # Keepa stores as 45 = 4.5 stars
        score += min(20, rating_stars * 4)
    
    # BSR (0-30 points) - lower is better
    csv = product.get("csv", [])
    bsr = 0
    if csv and len(csv) > 3 and csv[3]:
        bsr = csv[3][-1] if len(csv[3]) > 0 and csv[3][-1] != -1 else 0
    
    if bsr > 0:
        if bsr < 1000:
            score += 30
        elif bsr < 10000:
            score += 25
        elif bsr < 50000:
            score += 20
        elif bsr < 100000:
            score += 15
        elif bsr < 500000:
            score += 10
        else:
            score += 5
    
    # Amazon presence (0-20 points)
    availability = product.get("availabilityAmazon", -1)
    if availability == 0:  # In stock by Amazon
        score += 20
    elif availability == 1:  # Pre-order
        score += 15
    elif availability == 3:  # Backorder
        score += 10
    
    return score


def _is_valid_child(product: Dict) -> bool:
    """
    Check if a child variation is valid (not junk).
    
    Filters out:
    - Variation parents (productType=5) - these have NO data
    - Products with no price AND no BSR data
    - Products with very low ratings
    - Products with BSR > 500k (essentially dead)
    
    Args:
        product: Keepa product dictionary
        
    Returns:
        True if product passes quality filters
    """
    # CRITICAL: Exclude variation parents - they have NO usable data
    if _is_variation_parent(product):
        return False
    
    # Must have some usable data (price or BSR)
    if not _has_usable_data(product):
        return False
    
    # Extract metrics for additional filtering
    stats = product.get("stats", {})
    current = stats.get("current", []) if stats else []
    
    review_count = current[17] if len(current) > 17 and current[17] else 0
    rating = current[16] if len(current) > 16 and current[16] else 0
    
    csv = product.get("csv", [])
    bsr = 0
    if csv and len(csv) > 3 and csv[3]:
        bsr = csv[3][-1] if len(csv[3]) > 0 and csv[3][-1] != -1 else 0
    
    # Filter rules
    # 1. If has reviews, check rating (filter out low-rated products)
    if review_count > 0 and rating > 0:
        rating_stars = rating / 10
        if rating_stars < MIN_RATING:
            return False
    
    # 2. Filter dead products (very high BSR)
    if bsr > MAX_BSR:
        return False
    
    return True


# ============================================================================
# MAIN HARVESTER FUNCTIONS
# ============================================================================

def discover_seed_families(
    keyword: str,
    limit: int = DEFAULT_SEED_LIMIT,
    domain: str = "US",
    category_filter: Optional[int] = None,
    brand_filter: Optional[str] = None
) -> List[ProductFamily]:
    """
    Phase 1: Discover unique product families from keyword search.
    
    Uses singleVariation=true to get one product per family,
    avoiding duplicate variations in seed results.
    
    Args:
        keyword: Search term
        limit: Max number of seed families to find
        domain: Amazon marketplace
        category_filter: Optional category ID to restrict search
        brand_filter: Optional brand name to filter by
        
    Returns:
        List of ProductFamily objects (without children populated yet)
    """
    api_key = get_keepa_api_key()
    if not api_key:
        raise ValueError("KEEPA_API_KEY not found")
    
    # Build Product Finder query
    query_json = {
        "title": keyword,
        "perPage": max(50, limit * 2),  # Fetch extra to account for filtering
        "page": 0,
        "singleVariation": True,  # KEY: Get one product per family
        "current_SALES_gte": 1,
        "current_SALES_lte": 200000,  # Filter out dead products
        "sort": [["current_SALES", "asc"]]  # Best sellers first
    }
    
    # Optional filters
    if category_filter:
        query_json["rootCategory"] = [category_filter]
    
    if brand_filter:
        query_json["brand"] = [brand_filter]
    
    st.info(f"🔍 Searching for '{keyword}' with family-aware discovery...")
    
    # Execute search
    seed_asins = _keepa_product_finder_query(query_json, api_key, domain)
    
    if not seed_asins:
        st.warning(f"No products found for '{keyword}'")
        return []
    
    # Limit to requested amount
    seed_asins = seed_asins[:limit]
    
    st.info(f"📦 Found {len(seed_asins)} seed products, fetching details...")
    
    # Fetch full product details
    products = _fetch_product_details(seed_asins, api_key, domain)
    
    # Build ProductFamily objects
    # Key insight: Product Finder returns CHILD products (with data), not parent ASINs
    # Each child has a parentAsin field pointing to its parent (which has no data)
    families = []
    skipped_parents = 0
    
    for product in products:
        asin = product.get("asin", "")
        title = product.get("title", "Unknown")
        brand = product.get("brand", "Unknown")
        
        # Check if this is a variation parent (productType=5)
        # These have NO usable data (no price, no BSR) - skip them entirely
        is_var_parent = _is_variation_parent(product)
        has_data = _has_usable_data(product)
        
        if is_var_parent:
            skipped_parents += 1
            continue  # Skip variation parents - they have no data
        
        if not has_data:
            skipped_parents += 1
            continue  # Skip products with no usable data
        
        # Extract category
        root_category = product.get("rootCategory", 0)
        category_tree = product.get("categoryTree", [])
        category_path = " > ".join([cat.get("name", "") for cat in category_tree]) if category_tree else "Unknown"
        
        # Get BSR
        csv = product.get("csv", [])
        bsr = 0
        if csv and len(csv) > 3 and csv[3]:
            bsr = csv[3][-1] if len(csv[3]) > 0 and csv[3][-1] != -1 else 999999
        
        # Check if this product is itself a child (has a different parentAsin)
        parent_asin_from_product = product.get("parentAsin")
        is_child_of_another = parent_asin_from_product and parent_asin_from_product != asin
        
        # Extract any embedded variations (siblings)
        embedded_children = _extract_variation_asins(product)
        
        if is_child_of_another:
            # This product IS a child - treat the current ASIN as a child
            # Don't include the parent ASIN since we don't have its data
            family = ProductFamily(
                parent_asin=parent_asin_from_product,  # Reference to parent (no data)
                parent_title=title,
                parent_brand=brand,
                parent_bsr=bsr if bsr != 999999 else 0,
                child_asins=[asin] + [c for c in embedded_children if c != asin],  # Include self + siblings
                child_count=1 + len([c for c in embedded_children if c != asin]),
                is_variation_parent=False,
                category_id=root_category,
                category_path=category_path,
                has_parent_data=False  # Parent has no data - only children do
            )
        else:
            # This is a standalone product or the "hero" ASIN
            family = ProductFamily(
                parent_asin=asin,  # It IS the parent/hero
                parent_title=title,
                parent_brand=brand,
                parent_bsr=bsr if bsr != 999999 else 0,
                child_asins=embedded_children,
                child_count=len(embedded_children),
                is_variation_parent=False,
                category_id=root_category,
                category_path=category_path,
                has_parent_data=True  # This product has data
            )
        
        families.append(family)
    
    if skipped_parents > 0:
        st.caption(f"ℹ️ Filtered out {skipped_parents} products (no price/BSR data)")
    
    # Sort by BSR (best sellers first)
    families.sort(key=lambda f: f.parent_bsr)
    
    st.success(f"✅ Identified {len(families)} product families")
    return families


def explode_family_children(
    family: ProductFamily,
    api_key: str,
    domain: str = "US",
    max_children: int = MAX_CHILDREN_PER_PARENT
) -> ProductFamily:
    """
    Phase 2: Fetch ALL child variations for a product family.
    
    Uses historicalParentASIN filter to find all children of a parent.
    
    Args:
        family: ProductFamily object to populate
        api_key: Keepa API key
        domain: Amazon marketplace
        max_children: Safety cap on children per family
        
    Returns:
        Updated ProductFamily with all children populated
    """
    # If we already have children from variationCSV, use those
    if family.child_asins:
        st.caption(f"📋 {family.parent_title[:50]}... already has {len(family.child_asins)} variations cached")
        return family
    
    # Query for all historical children using historicalParentASIN
    query_json = {
        "historicalParentASIN": family.parent_asin,
        "perPage": max(50, max_children),
        "page": 0,
        "current_SALES_gte": 1,  # Must have some BSR
        "sort": [["current_SALES", "asc"]]  # Best sellers first
    }
    
    child_asins = _keepa_product_finder_query(query_json, api_key, domain)
    
    if child_asins:
        # Remove the parent if it's in the list
        child_asins = [a for a in child_asins if a != family.parent_asin]
        family.child_asins = child_asins[:max_children]
        family.child_count = len(family.child_asins)
        st.caption(f"🔗 {family.parent_title[:40]}... → {family.child_count} variations")
    
    return family


def filter_and_prioritize_children(
    family: ProductFamily,
    api_key: str,
    domain: str = "US"
) -> ProductFamily:
    """
    Phase 3: Filter junk children and prioritize by quality.
    
    Args:
        family: ProductFamily with children to filter
        api_key: Keepa API key
        domain: Amazon marketplace
        
    Returns:
        Updated ProductFamily with filtered/sorted children
    """
    if not family.child_asins:
        return family
    
    # Fetch details for all children
    products = _fetch_product_details(family.child_asins, api_key, domain)
    
    if not products:
        return family
    
    # Filter and score
    scored_children = []
    for product in products:
        if not _is_valid_child(product):
            continue
        
        asin = product.get("asin", "")
        score = _calculate_quality_score(product)
        scored_children.append((asin, score))
    
    # Sort by score (highest first) and extract ASINs
    scored_children.sort(key=lambda x: x[1], reverse=True)
    family.child_asins = [asin for asin, _ in scored_children]
    family.child_count = len(family.child_asins)
    
    return family


def harvest_product_families(
    keyword: str,
    max_asins: int = DEFAULT_TOTAL_QUOTA,
    domain: str = "US",
    category_filter: Optional[int] = None,
    brand_filter: Optional[str] = None,
    seed_limit: int = DEFAULT_SEED_LIMIT,
    expand_variations: bool = True,
    filter_children: bool = True
) -> HarvestResult:
    """
    Main entry point: Harvest complete product families for a keyword.
    
    This implements the full "Family Harvester" pattern:
    1. Search for seed products (unique families)
    2. Explode each family to get all variations
    3. Filter and prioritize children
    4. Fill quota with complete families
    
    Args:
        keyword: Search term (brand name, product type, etc.)
        max_asins: Maximum total ASINs to return
        domain: Amazon marketplace
        category_filter: Optional category ID to restrict search
        brand_filter: Optional brand name to filter by
        seed_limit: How many seed families to find initially
        expand_variations: Whether to explode variations (set False for speed)
        filter_children: Whether to apply quality filters to children
        
    Returns:
        HarvestResult with all ASINs, families, and stats
    """
    api_key = get_keepa_api_key()
    if not api_key:
        raise ValueError("KEEPA_API_KEY not found")
    
    # Phase 1: Discover seed families
    families = discover_seed_families(
        keyword=keyword,
        limit=seed_limit,
        domain=domain,
        category_filter=category_filter,
        brand_filter=brand_filter
    )
    
    if not families:
        return HarvestResult(
            asins=[],
            families=[],
            stats={"error": "No products found"}
        )
    
    # Phase 2 & 3: Explode and filter each family
    final_asins: List[str] = []
    final_families: List[ProductFamily] = []
    
    with st.expander("🔍 Family Expansion Progress", expanded=False):
        for family in families:
            if len(final_asins) >= max_asins:
                break
            
            # Explode variations
            if expand_variations:
                family = explode_family_children(family, api_key, domain)
            
            # Filter children
            if filter_children and family.child_asins:
                family = filter_and_prioritize_children(family, api_key, domain)
            
            # Calculate how many slots we have left
            slots_remaining = max_asins - len(final_asins)
            
            # Add this family's ASINs (up to remaining slots)
            family_asins = family.all_asins[:slots_remaining]
            final_asins.extend(family_asins)
            
            # Update family with actual included children
            if len(family_asins) < len(family.all_asins):
                # Truncated - update the family
                if family.is_variation_parent:
                    family.child_asins = family_asins
                else:
                    family.child_asins = family_asins[1:] if len(family_asins) > 1 else []
                family.child_count = len(family.child_asins)
            
            final_families.append(family)
            
            st.write(f"✅ {family.parent_brand} - {family.parent_title[:40]}... ({family.family_size} products)")
    
    # Build stats
    stats = {
        "keyword": keyword,
        "total_asins": len(final_asins),
        "total_families": len(final_families),
        "avg_family_size": np.mean([f.family_size for f in final_families]) if final_families else 0,
        "largest_family": max([f.family_size for f in final_families]) if final_families else 0,
        "brands_discovered": len(set(f.parent_brand for f in final_families))
    }
    
    st.success(
        f"🎯 Harvested {stats['total_asins']} ASINs across "
        f"{stats['total_families']} product families"
    )
    
    return HarvestResult(
        asins=final_asins,
        families=final_families,
        stats=stats
    )


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def harvest_to_seed_dataframe(
    keyword: str,
    limit: int = 50,
    domain: str = "US",
    category_filter: Optional[int] = None
) -> pd.DataFrame:
    """
    Compatibility function: Harvest families and return as seed DataFrame.
    
    This can be used as a drop-in replacement for phase1_seed_discovery()
    but with family-aware logic.
    
    Args:
        keyword: Search term
        limit: Max ASINs to return
        domain: Amazon marketplace
        category_filter: Optional category ID
        
    Returns:
        DataFrame compatible with phase1_seed_discovery format:
        [asin, title, brand, category_id, category_path, price, bsr]
    """
    result = harvest_product_families(
        keyword=keyword,
        max_asins=limit,
        domain=domain,
        category_filter=category_filter,
        seed_limit=min(10, limit // 5),  # Scale seeds based on limit
        expand_variations=True,
        filter_children=True
    )
    
    if not result.families:
        return pd.DataFrame()
    
    # Build DataFrame in phase1_seed_discovery format
    records = []
    for family in result.families:
        for asin in family.all_asins:
            records.append({
                "asin": asin,
                "title": family.parent_title,
                "brand": family.parent_brand,
                "category_id": family.category_id,
                "leaf_category_id": family.category_id,  # Will be refined later
                "category_tree_ids": [family.category_id],
                "category_tree_names": family.category_path.split(" > ") if family.category_path else [],
                "category_path": family.category_path,
                "price": 0,  # Will be filled in Phase 2
                "bsr": family.parent_bsr,
                "parent_asin": family.parent_asin,
                "family_size": family.family_size
            })
    
    df = pd.DataFrame(records)
    return df.drop_duplicates(subset=["asin"]).reset_index(drop=True)
