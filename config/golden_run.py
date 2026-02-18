"""
Golden Run Configuration
========================
Hardcodes a specific brand/seed for rapid iteration on the Market Misattribution Shield.

When GOLDEN_RUN_ENABLED = True:
- The discovery UI is bypassed entirely
- Session state is pre-populated with the golden brand's seed ASIN
- Market mapping runs against the golden category
- Use this during development to regenerate the brief without clicking through the UI

Usage:
    Set GOLDEN_RUN_ENABLED = True and fill in the brand details below.
    Then launch: streamlit run apps/shelfguard_app.py

    To reset and use the normal discovery UI, set GOLDEN_RUN_ENABLED = False.
"""

# ─── MASTER SWITCH ────────────────────────────────────────────────────────────
# Set True to bypass discovery UI and always run the golden brand
GOLDEN_RUN_ENABLED = False

# ─── GOLDEN BRAND DEFINITION ──────────────────────────────────────────────────
# Fill in your target brand + seed ASIN before enabling.

# The seed product that defines the market arena.
# Pick a well-ranked, representative ASIN from the brand (not a variation parent).
GOLDEN_SEED_ASIN = "B07XYZABC1"   # TODO: replace with your seed ASIN

# Brand name as it appears on Amazon listings (used to split portfolio vs competitors)
GOLDEN_BRAND = "CeraVe"           # TODO: replace with your golden brand

# Human-readable project name shown in the UI
GOLDEN_PROJECT_NAME = "CeraVe Skincare Arena"  # TODO: replace

# Keepa category ID for the market arena.
# Get this from the seed product's categoryTree after first discovery run.
# Skincare defaults: 11060451 (Skin Care), 11060691 (Face Moisturizers), 11060711 (Face Serums)
GOLDEN_CATEGORY_ID = 11060691     # TODO: replace with your category

GOLDEN_CATEGORY_NAME = "Face Moisturizers"   # TODO: replace

# Amazon domain (1=US, 2=UK, 3=DE, 4=FR, 5=JP, 6=CA, 7=IT, 8=ES)
GOLDEN_DOMAIN = 1

# Max ASINs to pull for the market map (100 = Keepa default, cap at 100)
GOLDEN_MARKET_SIZE = 100

# Category module ID (must match a key in src/workflow/config/__init__.py)
GOLDEN_CATEGORY_MODULE = "skincare_serum_moisturizer"

# ─── BRIEF CONTEXT ────────────────────────────────────────────────────────────
# Optional setup inputs surfaced in the brief (can be left as None)

# Whether the brand runs Sponsored Ads. If None → omit budget action language entirely.
# Set to True if you run ads, False if you don't, None if unknown.
GOLDEN_RUNS_ADS = None   # True | False | None

# Core SKUs for the brand (3–10 ASINs).
# If empty, the brief will determine core algorithmically from rank + revenue.
GOLDEN_CORE_SKUS = []    # e.g. ["B07XYZABC1", "B07XYZABC2"]


def get_golden_session_state() -> dict:
    """
    Return the session state dict that would normally be set by the discovery flow.
    Used to pre-populate session state when GOLDEN_RUN_ENABLED = True.
    """
    return {
        "active_project_asin": GOLDEN_SEED_ASIN,
        "active_project_name": GOLDEN_PROJECT_NAME,
        "active_project_seed_brand": GOLDEN_BRAND,
        "active_project_category_id": GOLDEN_CATEGORY_ID,
        "active_project_category_name": GOLDEN_CATEGORY_NAME,
        "active_project_all_asins": [],  # Populated after market map
        "category_module_id": GOLDEN_CATEGORY_MODULE,
        "golden_run_active": True,
        "golden_runs_ads": GOLDEN_RUNS_ADS,
        "golden_core_skus": GOLDEN_CORE_SKUS,
    }


def is_golden_run_active() -> bool:
    """Check whether the golden run is enabled and configured."""
    if not GOLDEN_RUN_ENABLED:
        return False
    if GOLDEN_SEED_ASIN.startswith("B07XYZ"):
        # Placeholder ASIN — not configured yet
        return False
    return True
