"""
LLM-Based Archetype Classifier.

Uses GPT to classify products into archetypes based on:
1. Deterministic features (from feature_extractor.py)
2. Product title and category
3. Portfolio context

Output is constrained to valid enum values.

Also includes rule-based fallback for when LLM is unavailable.
"""

import json
import os
from typing import Optional, Dict, Tuple
from datetime import datetime

from src.models.product_archetype import (
    ProductArchetype,
    CategoryArchetype,
    SKURole,
    ArchetypeFeatures,
)
from src.archetype.feature_extractor import (
    is_likely_fashion,
    is_likely_seasonal,
    is_likely_bundle,
    is_likely_consumable,
    is_likely_durable,
    is_likely_regulated,
    get_archetype_signals,
)


# =============================================================================
# LLM PROMPT TEMPLATE
# =============================================================================

CLASSIFIER_PROMPT = '''You are classifying an Amazon product into strategic categories.

=== PRODUCT INFO ===
ASIN: {asin}
Title: {title}
Category: {category}
Price: ${price:.2f}

=== EXTRACTED FEATURES ===
{features_json}

=== PORTFOLIO CONTEXT ===
{portfolio_context}

=== CLASSIFICATION TASK ===

**Step 1: Category Archetype**
Choose ONE archetype that best describes this product's economic model:

1. REPLENISHABLE_COMMODITY - Products bought repeatedly (toilet paper, batteries)
   - High purchase frequency, low loyalty, S&S important

2. RAZOR_AND_BLADES - Base + consumable model (printers/ink, razors/cartridges)
   - Base is loss leader, consumable is profit center

3. DURABLE_CONSIDERATION - High-price, infrequent purchase (furniture, appliances)
   - Research-heavy, reviews critical, returns expensive

4. SEASONAL_SPIKE - 80%+ revenue in <90 days (Halloween, Christmas items)
   - Timing is everything, clearance expected

5. REGULATED_COMPLIANCE - Requires trust/safety (supplements, baby products)
   - Unauthorized sellers are existential risk

6. FASHION_STYLE - Trend-driven, visual (clothing, accessories)
   - High returns normal, short lifecycle

7. COMMODITY_BUNDLE - Multipacks, variety packs, gift sets
   - Bundle economics differ from single unit

**Step 2: SKU Role**
Choose ONE role that best describes this SKU's strategic purpose:

1. ACQUISITION_HERO - Drive trial, capture new customers (lowest price, high ad spend)
2. PROFIT_ENGINE - Generate margin (highest margin, protect price)
3. DEFENDER - Block competitors (may not be profitable alone)
4. PROMO_LEVER - Drive traffic through deals (designed for discounting)
5. BUNDLE_LADDER - Increase AOV (multiple units or combos)
6. TRAFFIC_SPONGE - Capture keywords (high impression, lower conversion OK)
7. CLEANUP_LONGTAIL - Serve niche demand (minimal investment)

=== OUTPUT FORMAT ===
Return ONLY valid JSON:
{{
    "category_archetype": "REPLENISHABLE_COMMODITY" | "RAZOR_AND_BLADES" | "DURABLE_CONSIDERATION" | "SEASONAL_SPIKE" | "REGULATED_COMPLIANCE" | "FASHION_STYLE" | "COMMODITY_BUNDLE",
    "archetype_confidence": 0.0-1.0,
    "archetype_rationale": "Brief explanation (max 100 chars)",
    "sku_role": "ACQUISITION_HERO" | "PROFIT_ENGINE" | "DEFENDER" | "PROMO_LEVER" | "BUNDLE_LADDER" | "TRAFFIC_SPONGE" | "CLEANUP_LONGTAIL",
    "role_confidence": 0.0-1.0,
    "role_rationale": "Brief explanation (max 100 chars)"
}}
'''


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_product_archetype(
    asin: str,
    title: str,
    category: str,
    price: float,
    features: ArchetypeFeatures,
    portfolio_context: str = "",
    llm_client=None,
    use_llm: bool = True,
) -> ProductArchetype:
    """
    Classify a product into archetype and role.

    Uses LLM when available, falls back to rules otherwise.

    Args:
        asin: Product ASIN
        title: Product title
        category: Category tree/name
        price: Product price
        features: Extracted deterministic features
        portfolio_context: Context about other products in portfolio
        llm_client: OpenAI client (optional)
        use_llm: Whether to attempt LLM classification

    Returns:
        ProductArchetype with classification
    """
    # If LLM requested and client available, try LLM classification
    if use_llm and llm_client is not None:
        try:
            return _classify_with_llm(
                asin, title, category, price, features,
                portfolio_context, llm_client
            )
        except Exception as e:
            # Log error and fall back to rules
            print(f"LLM classification failed for {asin}: {e}")

    # Fall back to rule-based classification
    return _classify_with_rules(asin, title, category, price, features)


def _classify_with_llm(
    asin: str,
    title: str,
    category: str,
    price: float,
    features: ArchetypeFeatures,
    portfolio_context: str,
    llm_client,
) -> ProductArchetype:
    """
    Classify using LLM.

    Expects llm_client to be an OpenAI client instance.
    """
    # Build prompt
    prompt = CLASSIFIER_PROMPT.format(
        asin=asin,
        title=title,
        category=category,
        price=price,
        features_json=json.dumps(features.to_dict(), indent=2),
        portfolio_context=portfolio_context or "No portfolio context provided.",
    )

    # Call LLM
    response = llm_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    # Parse response into ProductArchetype
    try:
        archetype_value = result.get("category_archetype", "UNKNOWN").upper()
        role_value = result.get("sku_role", "UNKNOWN").upper()

        # Handle enum conversion
        category_archetype = CategoryArchetype.UNKNOWN
        if archetype_value in [a.name for a in CategoryArchetype]:
            category_archetype = CategoryArchetype[archetype_value]

        sku_role = SKURole.UNKNOWN
        if role_value in [r.name for r in SKURole]:
            sku_role = SKURole[role_value]

        return ProductArchetype(
            asin=asin,
            category_archetype=category_archetype,
            sku_role=sku_role,
            archetype_confidence=float(result.get("archetype_confidence", 0.5)),
            role_confidence=float(result.get("role_confidence", 0.5)),
            archetype_rationale=result.get("archetype_rationale", "LLM classification"),
            role_rationale=result.get("role_rationale", "LLM classification"),
            features=features,
            classified_at=datetime.now(),
        )

    except Exception as e:
        # If parsing fails, fall back to rules
        archetype = _classify_with_rules(asin, title, category, price, features)
        archetype.archetype_rationale = f"LLM parsing failed: {str(e)[:50]}"
        return archetype


def _classify_with_rules(
    asin: str,
    title: str,
    category: str,
    price: float,
    features: ArchetypeFeatures,
) -> ProductArchetype:
    """
    Rule-based fallback classification when LLM is unavailable.

    Uses the likelihood functions from feature_extractor.py.
    """
    # Get signal strengths for each archetype
    signals = get_archetype_signals(features, title)

    # Find the archetype with strongest signal
    archetype = CategoryArchetype.UNKNOWN
    archetype_conf = 0.0
    archetype_reason = "Rule-based classification"

    if signals:
        # Sort by signal strength
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        top_archetype, top_confidence = sorted_signals[0]

        if top_confidence > 0.3:
            archetype_map = {
                "replenishable_commodity": CategoryArchetype.REPLENISHABLE_COMMODITY,
                "razor_and_blades": CategoryArchetype.RAZOR_AND_BLADES,
                "durable_consideration": CategoryArchetype.DURABLE_CONSIDERATION,
                "seasonal_spike": CategoryArchetype.SEASONAL_SPIKE,
                "regulated_compliance": CategoryArchetype.REGULATED_COMPLIANCE,
                "fashion_style": CategoryArchetype.FASHION_STYLE,
                "commodity_bundle": CategoryArchetype.COMMODITY_BUNDLE,
            }
            archetype = archetype_map.get(top_archetype, CategoryArchetype.UNKNOWN)
            archetype_conf = top_confidence

            # Generate reason based on detected archetype
            reasons = {
                CategoryArchetype.REPLENISHABLE_COMMODITY: "Consumable product with subscription eligibility",
                CategoryArchetype.COMMODITY_BUNDLE: "Multipack or bundle product",
                CategoryArchetype.SEASONAL_SPIKE: "Seasonal keywords or high sales variance detected",
                CategoryArchetype.FASHION_STYLE: "Fashion/style keywords or size/color variations",
                CategoryArchetype.DURABLE_CONSIDERATION: "High price point or durable good keywords",
                CategoryArchetype.REGULATED_COMPLIANCE: "Health/safety regulated category",
                CategoryArchetype.RAZOR_AND_BLADES: "Consumable/refill pattern detected",
            }
            archetype_reason = reasons.get(archetype, "Rule-based classification")

    # Determine role based on price position and features
    role, role_conf, role_reason = _determine_sku_role(features, price)

    return ProductArchetype(
        asin=asin,
        category_archetype=archetype,
        sku_role=role,
        archetype_confidence=archetype_conf,
        role_confidence=role_conf,
        archetype_rationale=archetype_reason,
        role_rationale=role_reason,
        features=features,
        classified_at=datetime.now(),
    )


def _determine_sku_role(features: ArchetypeFeatures, price: float) -> Tuple[SKURole, float, str]:
    """
    Determine SKU role based on price position and features.

    Returns (role, confidence, reason).
    """
    # Bundle detection
    if features.is_multipack or features.pack_count > 1:
        return (
            SKURole.BUNDLE_LADDER,
            0.8,
            f"Multipack with {features.pack_count} units"
        )

    # Price position analysis
    price_ratio = features.price_vs_category_median

    if price_ratio < 0.7:
        # Below median = likely acquisition hero
        return (
            SKURole.ACQUISITION_HERO,
            0.7,
            "Price below category median (entry point)"
        )

    if price_ratio > 1.3:
        # Above median = likely profit engine
        return (
            SKURole.PROFIT_ENGINE,
            0.7,
            "Price above category median (premium positioning)"
        )

    # Low seller count + brand registry = could be defender
    if features.seller_count <= 3 and features.brand_registry_enrolled:
        return (
            SKURole.DEFENDER,
            0.5,
            "Low competition with brand protection"
        )

    # High seller count = traffic sponge (competitive keyword)
    if features.seller_count > 20:
        return (
            SKURole.TRAFFIC_SPONGE,
            0.5,
            "High competition keyword battleground"
        )

    # Low volume indicators = longtail
    if price < 10 and not features.has_subscribe_save:
        return (
            SKURole.CLEANUP_LONGTAIL,
            0.4,
            "Low price point without subscription value"
        )

    # Default to traffic sponge (generic)
    return (
        SKURole.TRAFFIC_SPONGE,
        0.3,
        "Default classification"
    )


# =============================================================================
# BATCH CLASSIFICATION
# =============================================================================

def classify_portfolio(
    products: list,
    llm_client=None,
    use_llm: bool = True,
) -> Dict[str, ProductArchetype]:
    """
    Classify all products in a portfolio.

    Args:
        products: List of dicts with product data (asin, title, category, price, features)
        llm_client: Optional OpenAI client
        use_llm: Whether to use LLM classification

    Returns:
        Dict mapping ASIN to ProductArchetype
    """
    results = {}

    # Build portfolio context
    portfolio_summary = f"Portfolio contains {len(products)} products"
    if products:
        categories = set(p.get("category", "")[:50] for p in products)
        portfolio_summary += f" in categories: {', '.join(list(categories)[:3])}"

    for product in products:
        asin = product.get("asin", "")
        if not asin:
            continue

        # Get or extract features
        features = product.get("features")
        if features is None:
            features = ArchetypeFeatures()

        archetype = classify_product_archetype(
            asin=asin,
            title=product.get("title", ""),
            category=product.get("category", ""),
            price=float(product.get("price", 0) or 0),
            features=features,
            portfolio_context=portfolio_summary,
            llm_client=llm_client,
            use_llm=use_llm,
        )

        results[asin] = archetype

    return results


# =============================================================================
# OVERRIDE FUNCTIONS
# =============================================================================

def override_archetype(
    archetype: ProductArchetype,
    new_category: Optional[CategoryArchetype] = None,
    new_role: Optional[SKURole] = None,
    override_by: str = "user",
) -> ProductArchetype:
    """
    Apply user override to an archetype classification.

    Args:
        archetype: The original classification
        new_category: New category archetype (if overriding)
        new_role: New SKU role (if overriding)
        override_by: Who made the override

    Returns:
        Updated ProductArchetype with override flag set
    """
    if new_category is not None:
        archetype.category_archetype = new_category
        archetype.archetype_confidence = 1.0
        archetype.archetype_rationale = f"User override by {override_by}"

    if new_role is not None:
        archetype.sku_role = new_role
        archetype.role_confidence = 1.0
        archetype.role_rationale = f"User override by {override_by}"

    if new_category is not None or new_role is not None:
        archetype.is_user_override = True
        archetype.override_timestamp = datetime.now()
        archetype.override_by = override_by

    return archetype
