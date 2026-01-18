"""
ShelfGuard Category Intelligence Engine
=========================================
Uses LLM to map user keywords to strategic categories and competitive sets.

Solves the "Infinite Set" problem by:
1. Mapping keywords to strategic categories (e.g., "Windex" → "Surface Cleaners")
2. Identifying 5-10 rival brands to define the competitive universe
3. Validating ASINs belong to the core category (not accessories/bundles)
"""

import json
from typing import Dict, List, Optional
import streamlit as st
from openai import OpenAI
import os


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client from secrets or environment."""
    try:
        # Try Streamlit secrets first
        return OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
    except:
        # Fallback to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None


@st.cache_data(ttl=86400)  # Cache for 24 hours
def map_keyword_to_category(keyword: str) -> Dict:
    """
    Use LLM to map a search keyword to a strategic category and competitive set.

    Args:
        keyword: User's search term (e.g., "Windex", "Starbucks K-Cups", "almond milk")

    Returns:
        Dict with:
        - category: Strategic category name
        - category_description: Brief description
        - rival_brands: List of 5-10 competing brands
        - exclude_keywords: Terms that indicate non-core products (accessories, bundles)
    """
    client = get_openai_client()

    if not client:
        # Fallback: return basic structure without LLM
        return {
            "category": f"{keyword} Products",
            "category_description": f"Products related to {keyword}",
            "rival_brands": [],
            "exclude_keywords": ["bundle", "pack of", "accessory", "replacement"]
        }

    prompt = f"""You are a market strategist helping define the competitive universe for "{keyword}".

Your task:
1. Identify the STRATEGIC CATEGORY this keyword belongs to (e.g., "Windex" → "Surface Cleaners", "Starbucks K-Cups" → "Single-Serve Coffee")
2. List 5-10 MAJOR RIVAL BRANDS that compete in this category
3. List keywords that indicate NON-CORE products (accessories, bundles, etc.) to exclude

Respond ONLY with valid JSON:
{{
  "category": "Strategic Category Name",
  "category_description": "Brief 1-sentence description of the category",
  "rival_brands": ["Brand1", "Brand2", "Brand3", ...],
  "exclude_keywords": ["bundle", "accessory", ...]
}}

Examples:

Input: "Windex"
Output:
{{
  "category": "Surface Cleaners",
  "category_description": "Household glass and multi-surface cleaning products",
  "rival_brands": ["Lysol", "Clorox", "Method", "Mrs. Meyer's", "Seventh Generation", "Pine-Sol", "Mr. Clean"],
  "exclude_keywords": ["bundle", "pack of", "refill", "sprayer", "bottle only", "nozzle", "dispenser"]
}}

Input: "Starbucks K-Cups"
Output:
{{
  "category": "Single-Serve Coffee Pods",
  "category_description": "Pre-portioned coffee pods for Keurig-compatible machines",
  "rival_brands": ["Dunkin'", "Green Mountain", "Peet's Coffee", "Newman's Own", "McCafe", "Folgers"],
  "exclude_keywords": ["machine", "brewer", "storage", "holder", "carousel"]
}}

Now process: "{keyword}"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a market intelligence expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for consistent categories
            max_tokens=500
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON
        result = json.loads(result_text)

        # Validate structure
        required_keys = ["category", "category_description", "rival_brands", "exclude_keywords"]
        if not all(key in result for key in required_keys):
            raise ValueError("Missing required keys in LLM response")

        return result

    except Exception as e:
        st.warning(f"⚠️ Category mapping failed: {str(e)}. Using basic fallback.")
        return {
            "category": f"{keyword} Products",
            "category_description": f"Products related to {keyword}",
            "rival_brands": [],
            "exclude_keywords": ["bundle", "pack of", "accessory", "replacement"]
        }


def validate_asin_relevance(
    product_title: str,
    category_context: Dict,
    original_keyword: str
) -> bool:
    """
    Validate if a product ASIN belongs to the core competitive category.

    Args:
        product_title: Product title from Keepa
        category_context: Output from map_keyword_to_category()
        original_keyword: User's original search term

    Returns:
        True if product is in-category, False if it's an accessory/bundle/irrelevant
    """
    title_lower = product_title.lower()

    # Exclude products with non-core keywords
    for exclude_term in category_context["exclude_keywords"]:
        if exclude_term.lower() in title_lower:
            return False

    # Must contain either the original keyword OR a rival brand
    valid_brands = [original_keyword.lower()] + [b.lower() for b in category_context["rival_brands"]]

    for brand in valid_brands:
        if brand in title_lower:
            return True

    # If no brand match, it's likely irrelevant
    return False


def expand_search_to_rivals(
    original_keyword: str,
    category_context: Dict,
    max_brands: int = 5
) -> List[str]:
    """
    Generate additional search keywords for rival brands.

    This allows us to define the FULL market (not just the searched brand).

    Args:
        original_keyword: User's search term
        category_context: Output from map_keyword_to_category()
        max_brands: Max number of rival brands to include

    Returns:
        List of search keywords (original + top rivals)
    """
    keywords = [original_keyword]

    # Add top N rival brands
    for rival in category_context["rival_brands"][:max_brands]:
        keywords.append(rival)

    return keywords


def calculate_market_share(
    brand_revenue: float,
    total_category_revenue: float
) -> float:
    """
    Calculate market share percentage with proper denominator.

    Args:
        brand_revenue: Revenue for the specific brand
        total_category_revenue: Total revenue across all brands in category

    Returns:
        Market share percentage (0-100)
    """
    if total_category_revenue == 0:
        return 0.0

    return (brand_revenue / total_category_revenue) * 100
