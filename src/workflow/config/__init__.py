"""
Category Module Configurations

Each category module defines:
- Detector thresholds (customized for category dynamics)
- Reason code weights (what's more important in this category)
- Action template mappings (category-specific actions)

V1: Skincare (serums + moisturizers) only.
Future modules will be added as config files here.
"""

from src.workflow.config.skincare_module import SKINCARE_CONFIG

# Registry of available category modules
CATEGORY_MODULES = {
    "skincare_serum_moisturizer": SKINCARE_CONFIG,
}

# Default module (used when no specific module is selected)
DEFAULT_MODULE_ID = "skincare_serum_moisturizer"


def get_category_config(module_id: str = None) -> dict:
    """
    Get category configuration by module ID.

    Args:
        module_id: Category module identifier. If None, returns default.

    Returns:
        Category configuration dict
    """
    if module_id is None:
        module_id = DEFAULT_MODULE_ID

    return CATEGORY_MODULES.get(module_id, SKINCARE_CONFIG)


def get_available_modules() -> list:
    """Get list of available category module IDs."""
    return list(CATEGORY_MODULES.keys())


def get_reason_code_weight(module_id: str, reason_code: str) -> float:
    """
    Get reason code weight for a specific category module.

    Args:
        module_id: Category module identifier
        reason_code: Reason code value (e.g., "price_war")

    Returns:
        Weight multiplier (1.0 = default)
    """
    config = get_category_config(module_id)
    weights = config.get("reason_code_weights", {})
    return weights.get(reason_code, 1.0)


def get_threshold(module_id: str, threshold_name: str) -> float:
    """
    Get threshold value for a specific category module.

    Args:
        module_id: Category module identifier
        threshold_name: Threshold name (e.g., "price_war_min_drops")

    Returns:
        Threshold value
    """
    config = get_category_config(module_id)
    thresholds = config.get("thresholds", {})
    return thresholds.get(threshold_name, 0.0)
