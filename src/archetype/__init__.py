"""
Archetype Classification Module

Provides context-aware product classification to enable
smarter signal interpretation.

The same signal means different things depending on:
1. Category Archetype: What kind of product economics does this have?
2. SKU Role: What strategic role does this SKU play in the portfolio?

Example:
- Price drop on Gillette razor handles (acquisition hero) = GOOD
- Price drop on Gillette cartridges (profit engine) = BAD

Key Components:
- feature_extractor: Extract deterministic features from product data
- classifier: LLM-based and rule-based classification
- playbooks: Signal interpretation rules per archetype
"""

from src.models.product_archetype import (
    CategoryArchetype,
    SKURole,
    ArchetypeFeatures,
    ProductArchetype,
)
from src.archetype.feature_extractor import (
    extract_archetype_features,
    is_likely_fashion,
    is_likely_seasonal,
    is_likely_bundle,
    is_likely_consumable,
    is_likely_durable,
    is_likely_regulated,
    get_archetype_signals,
)
from src.archetype.classifier import (
    classify_product_archetype,
    classify_portfolio,
    override_archetype,
)
from src.archetype.playbooks import (
    SignalType,
    SignalSentiment,
    SignalInterpretation,
    get_signal_interpretation,
    get_archetype_guidance,
    adjust_trigger_severity,
    ARCHETYPE_PLAYBOOKS,
    SKU_ROLE_MODIFIERS,
)

__all__ = [
    # Models
    "CategoryArchetype",
    "SKURole",
    "ArchetypeFeatures",
    "ProductArchetype",
    # Feature extraction
    "extract_archetype_features",
    "is_likely_fashion",
    "is_likely_seasonal",
    "is_likely_bundle",
    "is_likely_consumable",
    "is_likely_durable",
    "is_likely_regulated",
    "get_archetype_signals",
    # Classification
    "classify_product_archetype",
    "classify_portfolio",
    "override_archetype",
    # Playbooks
    "SignalType",
    "SignalSentiment",
    "SignalInterpretation",
    "get_signal_interpretation",
    "get_archetype_guidance",
    "adjust_trigger_severity",
    "ARCHETYPE_PLAYBOOKS",
    "SKU_ROLE_MODIFIERS",
]
