"""
Unit tests for Brief Round 3 additions:
    - build_baseline_signal() (features/regimes.py)
    - _build_secondary_signals() (report/weekly_brief.py)
    - ads_stance logic (features/asin_metrics.py)
    - Layer A curated receipts_list() (features/asin_metrics.py)
    - Baseline verdict in _build_misattribution_verdict() (report/weekly_brief.py)

Run with:
    pytest tests/test_brief_logic.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_df_weekly(asins_brands, n_weeks=2):
    """
    Build a minimal arena df_weekly with the required columns.

    asins_brands: list of (asin, brand, price, bsr_latest, bsr_prev)
    """
    rows = []
    today = datetime(2026, 2, 17)
    weeks = [today - timedelta(weeks=i) for i in reversed(range(n_weeks))]
    for asin, brand, price, bsr_latest, bsr_prev in asins_brands:
        for i, w in enumerate(weeks):
            bsr = bsr_prev if i == 0 else bsr_latest
            rows.append({
                "asin": asin,
                "brand": brand,
                "week_start": pd.Timestamp(w),
                "filled_price": price,
                "sales_rank_filled": bsr,
                "number_of_items": 1,
                "buy_box_price": price,
                "oos_pct_30": 0.0,
                "discount_days_7": 0,
            })
    return pd.DataFrame(rows)


def _make_metrics(
    asin="ASINTEST",
    brand="TestBrand",
    role="Core",
    tag="Stable leader",
    ad_waste_risk="Low",
    discount_persistence=0.0,
    price_vs_tier=0.0,
    price_vs_tier_band="in line",
    tier_comparable=True,
    bsr_wow=0.0,
    has_momentum=False,
    fidelity="weekly",
    ads_stance="Hold",
    product_type="other",
    family_id="",
    concerns=None,
    return_rate=None,
    sales_rank_drops_30=None,
    sales_rank_drops_90=None,
    monthly_sold_delta=None,
    top_comp_bb_share_30=None,
    has_buybox_stats=False,
    has_monthly_sold_history=False,
):
    from features.asin_metrics import ASINMetrics
    return ASINMetrics(
        asin=asin, brand=brand, role=role, tag=tag,
        ad_waste_risk=ad_waste_risk,
        discount_persistence=discount_persistence,
        price_vs_tier=price_vs_tier, price_vs_tier_band=price_vs_tier_band,
        tier_comparable=tier_comparable, bsr_wow=bsr_wow,
        has_momentum=has_momentum, fidelity=fidelity, ads_stance=ads_stance,
        product_type=product_type, family_id=family_id,
        concerns=concerns if concerns is not None else [],
        return_rate=return_rate,
        sales_rank_drops_30=sales_rank_drops_30,
        sales_rank_drops_90=sales_rank_drops_90,
        monthly_sold_delta=monthly_sold_delta,
        top_comp_bb_share_30=top_comp_bb_share_30,
        has_buybox_stats=has_buybox_stats,
        has_monthly_sold_history=has_monthly_sold_history,
    )


# ─── 1. build_baseline_signal ────────────────────────────────────────────────

class TestBuildBaselineSignal:
    def test_returns_baseline_regime(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(your_bsr_wow=0.05, arena_bsr_wow=0.03)
        assert sig.regime == "baseline"

    def test_active_is_true(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(0.05, 0.03)
        assert sig.active is True

    def test_exactly_two_receipts(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(0.05, 0.03)
        assert len(sig.receipts) == 2

    def test_driver_type_is_unknown_for_routing(self):
        """driver_type='Unknown' routes to tracking/divergence verdict logic, not Market-driven."""
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(0.05, 0.03)
        assert sig.driver_type == "Unknown"

    def test_confidence_defaults_to_med(self):
        """data_confidence defaults to 'Med' — not hardcoded 'Low'."""
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(0.05, 0.03)
        assert sig.confidence == "Med"

    def test_confidence_passthrough(self):
        """Caller can pass data_confidence to propagate actual data quality."""
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(0.05, 0.03, data_confidence="High")
        assert sig.confidence == "High"

    def test_verdict_text(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(0.05, 0.03)
        assert sig.verdict == "Baseline (No dominant market regime)"

    def test_arena_receipt_when_both_provided(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(your_bsr_wow=0.05, arena_bsr_wow=0.03)
        # First receipt should be arena-level
        assert "arena" in sig.receipts[0].label.lower() or "Arena" in sig.receipts[0].label

    def test_brand_vs_arena_receipt(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(your_bsr_wow=0.05, arena_bsr_wow=0.03)
        # Second receipt should be brand vs arena
        assert sig.receipts[1].metric == "brand_vs_arena_delta"

    def test_both_none_still_returns_two_receipts(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(your_bsr_wow=None, arena_bsr_wow=None)
        assert len(sig.receipts) == 2

    def test_only_arena_provided(self):
        from features.regimes import build_baseline_signal
        sig = build_baseline_signal(your_bsr_wow=None, arena_bsr_wow=0.03)
        assert len(sig.receipts) == 2
        assert "arena" in sig.receipts[0].label.lower() or "Arena" in sig.receipts[0].label

    def test_custom_band_fn(self):
        from features.regimes import build_baseline_signal
        band_fn = lambda v, t: "CUSTOM"
        sig = build_baseline_signal(0.05, 0.03, band_fn=band_fn)
        assert "CUSTOM" in sig.receipts[0].label


# ─── 2. Baseline verdict routing in _build_misattribution_verdict ─────────────

class TestBaselineVerdict:
    def _run_verdict(self, your_bsr, arena_bsr, fire_baseline=True):
        """Exercise _build_misattribution_verdict with a baseline signal."""
        from report.weekly_brief import _build_misattribution_verdict
        from features.regimes import build_baseline_signal
        firing = [build_baseline_signal(your_bsr, arena_bsr)] if fire_baseline else []
        return _build_misattribution_verdict(
            firing=firing,
            all_signals=[],
            conf_score=None,
            your_bsr=your_bsr,
            arena_bsr=arena_bsr,
        )

    def test_tracking_returns_baseline_verdict(self):
        """Brand tracking arena (delta ≤ 7pp) → 'Baseline (No dominant market regime)'."""
        verdict, conf, _ = self._run_verdict(your_bsr=0.05, arena_bsr=0.03)
        assert verdict == "Baseline (No dominant market regime)"

    def test_tracking_confidence_uses_data_confidence(self):
        """Baseline verdict_conf now uses conf_score.label, not hardcoded 'Low'.
        With conf_score=None passed, it falls back to 'Med'."""
        verdict, conf, _ = self._run_verdict(0.05, 0.03)
        # conf_score=None in test → fallback is "Med"
        assert conf == "Med"

    def test_diverging_returns_unknown(self):
        """Brand diverging from arena (delta > 7pp) → 'Unknown'."""
        verdict, conf, _ = self._run_verdict(your_bsr=0.20, arena_bsr=0.03)
        assert verdict == "Unknown"

    def test_baseline_receipts_not_empty(self):
        _, _, receipts = self._run_verdict(0.05, 0.03)
        assert len(receipts) == 2
        assert all(r for r in receipts)  # no empty strings


# ─── 3. _build_secondary_signals ─────────────────────────────────────────────

class TestSecondarySignals:
    def _run(self, price_vs_tier, asin_metrics, your_brand, regime_names=None):
        from report.weekly_brief import _build_secondary_signals

        # Build fake firing list with given regime names
        class _FakeSig:
            def __init__(self, regime):
                self.regime = regime

        firing = [_FakeSig(r) for r in (regime_names or [])]
        band_fn = lambda v, t: f"{v*100:+.1f}%"
        return _build_secondary_signals(price_vs_tier, asin_metrics, your_brand, firing, band_fn)

    def test_price_pressure_above_threshold(self):
        signals = self._run(price_vs_tier=0.10, asin_metrics={}, your_brand="X")
        assert any("Price pressure" in s.claim for s in signals)

    def test_price_pressure_has_two_receipts(self):
        signals = self._run(price_vs_tier=0.10, asin_metrics={}, your_brand="X")
        price_signals = [s for s in signals if "Price pressure" in s.claim]
        assert price_signals
        assert len(price_signals[0].receipts) == 2

    def test_price_pressure_below_threshold_not_included(self):
        signals = self._run(price_vs_tier=0.03, asin_metrics={}, your_brand="X")
        assert not any("Price pressure" in s.claim for s in signals)

    def test_price_pressure_none_not_included(self):
        signals = self._run(price_vs_tier=None, asin_metrics={}, your_brand="X")
        assert not any("Price pressure" in s.claim for s in signals)

    def test_price_pressure_suppressed_when_tier_compression_active(self):
        signals = self._run(0.10, {}, "X", regime_names=["tier_compression"])
        assert not any("Price pressure" in s.claim for s in signals)

    def test_price_pressure_suppressed_when_promo_war_active(self):
        signals = self._run(0.10, {}, "X", regime_names=["promo_war"])
        assert not any("Price pressure" in s.claim for s in signals)

    def test_discount_concentration_above_threshold(self):
        asin_metrics = {
            "A1": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
            "A2": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
            "A3": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
            "A4": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
            "A5": _make_metrics(role="Core", discount_persistence=0.1),
        }
        # 4 out of 5 = 80% ≥ 40%
        signals = self._run(None, asin_metrics, "X")
        assert any("Discount concentration" in s.claim for s in signals)

    def test_discount_concentration_has_two_receipts(self):
        asin_metrics = {
            "A1": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
            "A2": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
        }
        signals = self._run(None, asin_metrics, "X")
        disc_signals = [s for s in signals if "Discount concentration" in s.claim]
        assert disc_signals
        assert len(disc_signals[0].receipts) == 2

    def test_discount_concentration_below_threshold(self):
        asin_metrics = {
            "A1": _make_metrics(role="Core", discount_persistence=0.5),
            "A2": _make_metrics(role="Core", discount_persistence=0.0),
            "A3": _make_metrics(role="Core", discount_persistence=0.0),
            "A4": _make_metrics(role="Core", discount_persistence=0.0),
        }
        # 1 out of 4 = 25% < 40%
        signals = self._run(None, asin_metrics, "X")
        assert not any("Discount concentration" in s.claim for s in signals)

    def test_discount_concentration_suppressed_when_promo_war_active(self):
        asin_metrics = {
            "A1": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
            "A2": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
        }
        signals = self._run(None, asin_metrics, "X", regime_names=["promo_war"])
        assert not any("Discount concentration" in s.claim for s in signals)

    def test_max_two_signals(self):
        asin_metrics = {
            "A1": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
            "A2": _make_metrics(role="Core", discount_persistence=4/7 + 0.01),
        }
        signals = self._run(0.10, asin_metrics, "X")
        assert len(signals) <= 2


# ─── 4. Ads stance logic in compute_asin_metrics ─────────────────────────────

class TestAdsStance:
    """
    Tests for ads_stance logic in compute_asin_metrics().

    We exercise the logic directly via the ASINMetrics dataclass fields and
    the standalone logic, since compute_asin_metrics() requires a well-formed
    df_weekly. For integration-level tests we build a minimal df_weekly.
    """

    def _compute(self, price_vs_tier, bsr_wow, bsr_wow_prev=0.0,
                 competitor_pressure=False, tier_comparable=True,
                 brand="TestBrand", your_brand="TestBrand"):
        """Build a minimal df_weekly and run compute_asin_metrics()."""
        from features.asin_metrics import compute_asin_metrics

        # bsr_prev → bsr_latest derived from bsr_wow
        bsr_prev = 10000
        bsr_latest = int(bsr_prev * (1 + bsr_wow)) if bsr_wow is not None else bsr_prev
        price = 20.0
        tier_price = price / (1 + price_vs_tier) if tier_comparable else price

        # Build 3 ASINs so tier_median is meaningful
        data = [
            ("ASIN001", brand, price, bsr_latest, bsr_prev),
            ("ASIN002", "CompBrand", tier_price, bsr_prev, bsr_prev),
            ("ASIN003", "CompBrand", tier_price, bsr_prev, bsr_prev),
        ]
        df = _make_df_weekly(data, n_weeks=2)

        if not tier_comparable:
            # Make items count extreme so it's excluded from comparable set
            df.loc[df["asin"] == "ASIN001", "number_of_items"] = 999

        results = compute_asin_metrics(
            df, {}, {}, your_brand=your_brand,
            competitor_pressure=competitor_pressure,
        )
        return results.get("ASIN001")

    def test_not_tier_comparable_returns_hold(self):
        m = self._compute(price_vs_tier=0.0, bsr_wow=0.0, tier_comparable=False)
        assert m is not None
        assert m.ads_stance == "Hold"

    def test_competitive_significant_gain_returns_scale(self):
        """Price ≥5% below tier (is_competitive), BSR improved > 10% → Scale."""
        # is_competitive threshold: price_vs_tier <= -0.05 (default competitive_price_below_tier_pct)
        m = self._compute(price_vs_tier=-0.10, bsr_wow=-0.15)
        assert m is not None
        assert m.ads_stance == "Scale"

    def test_competitive_flat_returns_defend(self):
        """Price ≥5% below tier (is_competitive), BSR flat → Defend."""
        m = self._compute(price_vs_tier=-0.10, bsr_wow=0.00)
        assert m is not None
        assert m.ads_stance == "Defend"

    def test_not_competitive_deteriorating_with_pressure_returns_pause(self):
        """Price above tier (not competitive), BSR ≥20% worse, competitor_pressure=True → Pause+Diagnose."""
        # is_deteriorating threshold: bsr_wow >= 0.20 (default bsr_deterioration_pct)
        m = self._compute(price_vs_tier=0.20, bsr_wow=0.25, competitor_pressure=True)
        assert m is not None
        assert m.ads_stance == "Pause+Diagnose"

    def test_not_competitive_deteriorating_without_pressure_returns_hold(self):
        """Same conditions without competitor_pressure → Hold (not Pause+Diagnose)."""
        m = self._compute(price_vs_tier=0.20, bsr_wow=0.25, competitor_pressure=False)
        assert m is not None
        assert m.ads_stance in ("Hold", "Defend")  # no Pause+Diagnose without pressure


# ─── 5. Layer A receipts_list curation ───────────────────────────────────────

class TestReceiptsList:
    def _make_metrics_dict(self, brand="MyBrand"):
        """
        Create a set of metrics with both Pause+Diagnose and Scale stances
        for the brand, plus a competitor.
        """
        from features.asin_metrics import ASINMetrics
        return {
            "MINE01": _make_metrics("MINE01", brand, role="Core", bsr_wow=0.20,
                                     ads_stance="Pause+Diagnose"),
            "MINE02": _make_metrics("MINE02", brand, role="Core", bsr_wow=-0.15,
                                     ads_stance="Scale"),
            "MINE03": _make_metrics("MINE03", brand, role="Core", bsr_wow=0.10,
                                     ads_stance="Hold"),
            "COMP01": _make_metrics("COMP01", "CompBrand", role="Core", bsr_wow=0.25,
                                     ads_stance="Hold"),
        }

    def test_returns_list_of_strings(self):
        from features.asin_metrics import receipts_list
        metrics = self._make_metrics_dict()
        result = receipts_list(metrics, "MyBrand", max_items=6)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_contains_pause_diagnose_skus(self):
        from features.asin_metrics import receipts_list
        metrics = self._make_metrics_dict()
        result = receipts_list(metrics, "MyBrand", max_items=6)
        # MINE01 is Pause+Diagnose with high |bsr_wow| — should appear
        combined = " ".join(result)
        assert "MINE01"[-6:] in combined

    def test_contains_scale_skus(self):
        from features.asin_metrics import receipts_list
        metrics = self._make_metrics_dict()
        result = receipts_list(metrics, "MyBrand", max_items=6)
        combined = " ".join(result)
        assert "MINE02"[-6:] in combined

    def test_max_items_respected(self):
        from features.asin_metrics import receipts_list
        metrics = self._make_metrics_dict()
        result = receipts_list(metrics, "MyBrand", max_items=3)
        assert len(result) <= 3

    def test_fallback_when_no_brand_asins(self):
        """When brand has no ASINs, falls back to |bsr_wow| sort."""
        from features.asin_metrics import receipts_list
        metrics = {
            "COMP01": _make_metrics("COMP01", "CompBrand", bsr_wow=0.30),
            "COMP02": _make_metrics("COMP02", "CompBrand", bsr_wow=0.10),
        }
        result = receipts_list(metrics, "UnknownBrand", max_items=6)
        assert len(result) > 0

    def test_ads_hint_in_non_hold_stance(self):
        """SKUs with non-Hold stance should include ads hint in receipt line."""
        from features.asin_metrics import receipts_list
        metrics = {
            "MINE01": _make_metrics("MINE01", "MyBrand", bsr_wow=0.20,
                                     ads_stance="Pause+Diagnose"),
        }
        result = receipts_list(metrics, "MyBrand", max_items=6)
        combined = " ".join(result)
        assert "Pause+Diagnose" in combined or "ads" in combined.lower()


# ─── 7. Taxonomy — classify_title + infer_module + group metrics ──────────────

class TestTaxonomy:
    """Tests for Round 5 taxonomy: classify_title(), infer_module(), compute_group_metrics()."""

    # classify_title -----------------------------------------------------------

    def test_classify_serum(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("CeraVe Hyaluronic Acid Serum for Face") == "serum"

    def test_classify_moisturizer_from_cream(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Neutrogena Hydro Boost Face Cream") == "moisturizer"

    def test_classify_cleanser(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Cetaphil Gentle Skin Cleanser 16 fl oz") == "cleanser"

    def test_classify_face_wash(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("CeraVe Foaming Face Wash Facial Cleanser") == "cleanser"

    def test_classify_sunscreen(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("EltaMD UV Clear Broad-Spectrum SPF 46") == "sunscreen"

    def test_classify_body_wash(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Dove Sensitive Skin Body Wash") == "body wash"

    def test_classify_retinol(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("RoC Retinol Correxion Line Smoothing Serum") == "retinol"

    def test_classify_other(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Some Random Nutrition Supplement") == "other"

    def test_classify_empty_string(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("") == "other"

    def test_classify_case_insensitive(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("VITAMIN C SERUM 20% FACE BRIGHTENING") == "serum"

    # infer_module -------------------------------------------------------------

    def test_infer_skincare_from_skin_care(self):
        from config.market_misattribution_module import infer_module
        assert infer_module("Health & Household > Skin Care > Face Moisturizers") == "skincare"

    def test_infer_skincare_from_serum(self):
        from config.market_misattribution_module import infer_module
        assert infer_module("Beauty > Face Serum Products") == "skincare"

    def test_infer_generic_grocery(self):
        from config.market_misattribution_module import infer_module
        assert infer_module("Grocery & Gourmet Food > Beverages > Soft Drinks") == "generic"

    def test_infer_generic_empty(self):
        from config.market_misattribution_module import infer_module
        assert infer_module("") == "generic"

    # ASINMetrics has product_type field ---------------------------------------

    def test_asin_metrics_has_product_type_default(self):
        """New product_type field defaults to 'other' — backward compatible."""
        m = _make_metrics("B01", "TestBrand")
        assert hasattr(m, "product_type")
        assert m.product_type == "other"

    def test_asin_metrics_has_family_id_default(self):
        """New family_id field defaults to '' — backward compatible."""
        m = _make_metrics("B01", "TestBrand")
        assert hasattr(m, "family_id")
        assert m.family_id == ""

    # compute_asin_metrics populates product_type via classify_title -----------

    def test_compute_asin_metrics_populates_product_type(self):
        """compute_asin_metrics() reads title and classifies it."""
        df = _make_df_weekly([("B01", "Nat", 25.0, 3000, 3100)])
        # Add title column so classify_title can fire
        df["title"] = "Naturium Vitamin C Complex Serum"
        from features.asin_metrics import compute_asin_metrics
        metrics = compute_asin_metrics(df, {}, {}, "Nat")
        assert "B01" in metrics
        assert metrics["B01"].product_type == "serum"

    # compute_group_metrics ----------------------------------------------------

    def test_compute_group_metrics_groups_by_product_type_and_brand(self):
        """Two ASINs with different product_types produce two separate groups."""
        m1 = _make_metrics("B01", "Nat", bsr_wow=0.0, product_type="serum")
        m2 = _make_metrics("B02", "Nat", bsr_wow=0.0, product_type="moisturizer")
        df = _make_df_weekly([
            ("B01", "Nat", 25.0, 3000, 3100),
            ("B02", "Nat", 18.0, 4000, 4100),
        ])
        from features.asin_metrics import compute_group_metrics
        groups = compute_group_metrics({"B01": m1, "B02": m2}, df, "Nat")
        ptypes = {g.product_type for g in groups}
        assert "serum" in ptypes
        assert "moisturizer" in ptypes

    def test_compute_group_metrics_your_brand_first(self):
        """Your brand groups appear before competitor groups."""
        m1 = _make_metrics("B01", "Nat", bsr_wow=0.0, product_type="serum")
        m2 = _make_metrics("B02", "Comp", bsr_wow=0.0, product_type="serum")
        df = _make_df_weekly([
            ("B01", "Nat", 25.0, 1000, 1100),
            ("B02", "Comp", 20.0, 800, 850),
        ])
        from features.asin_metrics import compute_group_metrics
        groups = compute_group_metrics({"B01": m1, "B02": m2}, df, "Nat")
        assert groups[0].brand == "Nat"

    def test_compute_group_metrics_momentum_gaining(self):
        """Group with majority has_momentum=True → momentum_label='gaining'."""
        m1 = _make_metrics("B01", "Nat", bsr_wow=-0.15, has_momentum=True, product_type="serum")
        m2 = _make_metrics("B02", "Nat", bsr_wow=-0.12, has_momentum=True, product_type="serum")
        df = _make_df_weekly([
            ("B01", "Nat", 25.0, 2000, 2400),
            ("B02", "Nat", 25.0, 2500, 2900),
        ])
        from features.asin_metrics import compute_group_metrics
        groups = compute_group_metrics({"B01": m1, "B02": m2}, df, "Nat")
        serum_group = next(g for g in groups if g.product_type == "serum")
        assert serum_group.momentum_label == "gaining"

    # to_group_table -----------------------------------------------------------

    def test_to_group_table_returns_dataframe(self):
        from features.asin_metrics import ProductGroupMetrics, to_group_table
        g = ProductGroupMetrics(
            product_type="serum", brand="Nat", asin_count=3,
            rev_share_pct=0.24, median_price_vs_tier=0.05, pct_discounted=0.2,
            momentum_label="gaining", dominant_ads_stance="Scale", top_asins=["B01"],
        )
        df = to_group_table([g])
        assert not df.empty
        assert "Product Type" in df.columns
        assert "Brand" in df.columns
        assert df.iloc[0]["Product Type"] == "Serum"


# ─── 8. Round 6 — Posture table, Actions block, Curated Layer A ───────────────

class TestRound6:
    """Tests for Round 6: posture display layer, compact table columns,
    curated receipts, validate notes, actions block, baseline headline."""

    # _POSTURE_DISPLAY / _POSTURE_NEXT_ACTION / _VALIDATE_BY_STANCE ──────────

    def test_posture_display_hold(self):
        from features.asin_metrics import _POSTURE_DISPLAY
        assert _POSTURE_DISPLAY["Hold"] == "Hold budget"

    def test_posture_display_pause(self):
        from features.asin_metrics import _POSTURE_DISPLAY
        assert _POSTURE_DISPLAY["Pause+Diagnose"] == "Pause incremental"

    def test_posture_display_scale(self):
        from features.asin_metrics import _POSTURE_DISPLAY
        assert _POSTURE_DISPLAY["Scale"] == "Scale"

    def test_posture_display_defend(self):
        from features.asin_metrics import _POSTURE_DISPLAY
        assert _POSTURE_DISPLAY["Defend"] == "Defend"

    def test_validate_by_stance_keys(self):
        from features.asin_metrics import _VALIDATE_BY_STANCE
        assert "Scale" in _VALIDATE_BY_STANCE
        assert "Pause+Diagnose" in _VALIDATE_BY_STANCE
        assert "Hold" in _VALIDATE_BY_STANCE
        assert "Defend" in _VALIDATE_BY_STANCE

    # to_compact_table columns ────────────────────────────────────────────────

    def test_compact_table_has_posture_column(self):
        from features.asin_metrics import to_compact_table
        metrics = {
            "ASIN01": _make_metrics("ASIN01", "Brand", ads_stance="Scale"),
            "ASIN02": _make_metrics("ASIN02", "Brand", ads_stance="Hold"),
        }
        df = _make_df_weekly([("ASIN01", "Brand", 25.0, 2000, 2100),
                              ("ASIN02", "Brand", 25.0, 3000, 3100)])
        result = to_compact_table(metrics, df)
        assert "Posture" in result.columns
        assert "Next action" in result.columns
        assert "Ads stance" not in result.columns

    def test_compact_table_posture_values(self):
        from features.asin_metrics import to_compact_table
        metrics = {
            "A01": _make_metrics("A01", "Brand", ads_stance="Scale"),
            "A02": _make_metrics("A02", "Brand", ads_stance="Hold"),
            "A03": _make_metrics("A03", "Brand", ads_stance="Pause+Diagnose"),
        }
        df = _make_df_weekly([("A01", "Brand", 25.0, 2000, 2100),
                              ("A02", "Brand", 25.0, 2500, 2600),
                              ("A03", "Brand", 25.0, 3000, 3100)])
        result = to_compact_table(metrics, df)
        postures = set(result["Posture"].tolist())
        assert "Scale" in postures
        assert "Hold budget" in postures
        assert "Pause incremental" in postures

    # receipts_list curation order ────────────────────────────────────────────

    def test_receipts_list_curates_scale_first(self):
        """Scale SKU should appear before Pause+Diagnose in Layer A."""
        from features.asin_metrics import receipts_list
        metrics = {
            "SCALE1": _make_metrics("SCALE1", "MyBrand", bsr_wow=-0.20,
                                    ads_stance="Scale"),
            "PAUSE1": _make_metrics("PAUSE1", "MyBrand", bsr_wow=0.25,
                                    ads_stance="Pause+Diagnose"),
        }
        result = receipts_list(metrics, "MyBrand", max_items=6)
        assert len(result) == 2
        # Scale line should come first
        assert "SCALE1"[-6:] in result[0]
        assert "PAUSE1"[-6:] in result[1]

    def test_receipts_list_validate_note_when_ads_true(self):
        """runs_ads=True → each receipt line contains 'Validate:'."""
        from features.asin_metrics import receipts_list
        metrics = {
            "MINE01": _make_metrics("MINE01", "MyBrand", bsr_wow=-0.15,
                                    ads_stance="Scale"),
        }
        result = receipts_list(metrics, "MyBrand", max_items=6, runs_ads=True)
        assert len(result) == 1
        assert "Validate:" in result[0]

    def test_receipts_list_validate_note_when_ads_none(self):
        """runs_ads=None (unknown) → validate hint still included."""
        from features.asin_metrics import receipts_list
        metrics = {
            "MINE01": _make_metrics("MINE01", "MyBrand", bsr_wow=-0.15,
                                    ads_stance="Defend"),
        }
        result = receipts_list(metrics, "MyBrand", max_items=6, runs_ads=None)
        assert "Validate:" in result[0]

    def test_receipts_list_no_validate_when_ads_false(self):
        """runs_ads=False → no 'Validate:' in any receipt."""
        from features.asin_metrics import receipts_list
        metrics = {
            "MINE01": _make_metrics("MINE01", "MyBrand", bsr_wow=0.20,
                                    ads_stance="Pause+Diagnose"),
            "MINE02": _make_metrics("MINE02", "MyBrand", bsr_wow=-0.10,
                                    ads_stance="Scale"),
        }
        result = receipts_list(metrics, "MyBrand", max_items=6, runs_ads=False)
        combined = " ".join(result)
        assert "Validate:" not in combined

    # _build_actions_block ─────────────────────────────────────────────────────

    def test_actions_block_baseline_posture(self):
        """Baseline week → budget posture bullet says 'Hold total budget'."""
        from report.weekly_brief import _build_actions_block
        from features.regimes import RegimeSignal
        baseline_signal = RegimeSignal(
            regime="baseline", active=True, confidence="High",
            verdict="No dominant regime", driver_type="Unknown",
        )
        metrics = {
            "A01": _make_metrics("A01", "MyBrand", ads_stance="Hold"),
        }
        result = _build_actions_block(
            [baseline_signal], metrics, "MyBrand", "Hold", None,
        )
        assert len(result) == 3
        assert "Hold total budget" in result[0]

    def test_actions_block_reallocation_when_pause_and_scale(self):
        """Pause+Diagnose + Scale SKUs → reallocation bullet names both sets."""
        from report.weekly_brief import _build_actions_block
        from features.regimes import RegimeSignal
        baseline = RegimeSignal(
            regime="baseline", active=True, confidence="Med",
            verdict="No regime", driver_type="Unknown",
        )
        metrics = {
            "P001": _make_metrics("P001", "MyBrand", ads_stance="Pause+Diagnose"),
            "S001": _make_metrics("S001", "MyBrand", ads_stance="Scale"),
        }
        result = _build_actions_block(
            [baseline], metrics, "MyBrand", "Reallocate", True,
        )
        realloc_line = result[1]
        assert "P001"[-6:] in realloc_line
        assert "S001"[-6:] in realloc_line
        assert "Reallocation" in realloc_line

    def test_actions_block_no_comp_pressure(self):
        """No discounting+gaining comps → competitor pressure bullet says 'No discounting'."""
        from report.weekly_brief import _build_actions_block
        from features.regimes import RegimeSignal
        sig = RegimeSignal(
            regime="baseline", active=True, confidence="High",
            verdict="No regime", driver_type="Unknown",
        )
        metrics = {
            "A01": _make_metrics("A01", "MyBrand", ads_stance="Hold",
                                  discount_persistence=0.0, bsr_wow=0.05),
        }
        result = _build_actions_block([sig], metrics, "MyBrand", "Hold", None)
        assert "No discounting" in result[2]

    # Baseline headline ────────────────────────────────────────────────────────

    def test_headline_baseline_language(self):
        """baseline regime → headline contains 'No dominant market regime'."""
        from report.weekly_brief import _build_headline
        from features.regimes import RegimeSignal
        from scoring.confidence import ConfidenceScore
        baseline = RegimeSignal(
            regime="baseline", active=True, confidence="High",
            verdict="No regime", driver_type="Unknown",
        )
        conf = ConfidenceScore(label="High", score=2, data_quality="Good",
                               arena_coverage=0.8, reasons=[])
        headline = _build_headline("CeraVe", -0.01, -0.02, [baseline], conf)
        assert "No dominant market regime" in headline
        assert "CeraVe" in headline
        assert "High" in headline


# ─── 9. Round 7: universal bucket language ────────────────────────────────────

class TestRound7:
    """Round 7: universal bucket language, new classify_title terms,
    bucket What Changed, bucket reallocation in Actions, group-first Layer B."""

    # New classify_title terms ─────────────────────────────────────────────────

    def test_classify_niacinamide_returns_serum(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Niacinamide 12% Solution") == "serum"

    def test_classify_vitamin_c_returns_serum(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Vitamin C Brightening Serum 20%") == "serum"

    def test_classify_glycolic_returns_exfoliant(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Glycolic Acid 7% Toning Solution") == "exfoliant"

    def test_classify_salicylic_returns_exfoliant(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Salicylic Acid 0.5% Daily Toner") == "exfoliant"

    def test_classify_acne_returns_acne_treatment(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("CeraVe Acne Foaming Cream Wash") == "acne_treatment"

    def test_classify_benzoyl_returns_acne_treatment(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Benzoyl Peroxide 2.5% Face Wash") == "acne_treatment"

    def test_classify_spot_treat_returns_acne_treatment(self):
        from config.market_misattribution_module import classify_title
        assert classify_title("Kate Somerville EradiKate Spot Treatment Acne") == "acne_treatment"

    # _has_buckets ─────────────────────────────────────────────────────────────

    def test_has_buckets_true_when_non_other(self):
        from report.weekly_brief import _has_buckets
        from features.asin_metrics import ProductGroupMetrics
        g = ProductGroupMetrics("serum", "Brand", 2, 0.1, 0.0, 0.2, "gaining", "Scale", [])
        assert _has_buckets([g]) is True

    def test_has_buckets_false_when_all_other(self):
        from report.weekly_brief import _has_buckets
        from features.asin_metrics import ProductGroupMetrics
        g = ProductGroupMetrics("other", "Brand", 2, 0.1, 0.0, 0.2, "flat", "Hold", [])
        assert _has_buckets([g]) is False

    def test_has_buckets_false_when_empty(self):
        from report.weekly_brief import _has_buckets
        assert _has_buckets([]) is False
        assert _has_buckets(None) is False

    # _build_headline without skincare gate ────────────────────────────────────

    def test_headline_bucket_note_generic_module(self):
        """module_id='generic' + non-other group → bucket note still appears."""
        from report.weekly_brief import _build_headline
        from features.regimes import RegimeSignal
        from scoring.confidence import ConfidenceScore
        from features.asin_metrics import ProductGroupMetrics
        sig = RegimeSignal(regime="promo_war", active=True, confidence="High",
                           verdict="Promo war active", driver_type="Market-driven")
        conf = ConfidenceScore(label="High", score=2, data_quality="Good",
                               arena_coverage=0.8, reasons=[])
        g = ProductGroupMetrics("serum", "CeraVe", 3, 0.30, 0.0, 0.2, "gaining", "Scale", [])
        headline = _build_headline("CeraVe", -0.05, -0.02, [sig], conf,
                                   module_id="generic", group_metrics=[g])
        assert "Serum" in headline

    # _build_what_changed bucket-first ────────────────────────────────────────

    def test_what_changed_brand_bucket_bullet(self):
        """When group_metrics has brand serum group → first bullet names bucket."""
        from report.weekly_brief import _build_what_changed
        from features.asin_metrics import ProductGroupMetrics
        g = ProductGroupMetrics("serum", "MyBrand", 3, 0.30, 0.0, 0.2, "gaining", "Scale", [])
        bullets = _build_what_changed(
            -0.05, -0.02, None, None, [], lambda v, t: f"{v*100:+.1f}%",
            your_brand="MyBrand", group_metrics=[g],
        )
        combined = " ".join(bullets)
        assert "Serum" in combined

    def test_what_changed_fallback_when_no_buckets(self):
        """No group_metrics → standard arena demand bullet."""
        from report.weekly_brief import _build_what_changed
        bullets = _build_what_changed(
            -0.05, -0.04, None, None, [], lambda v, t: f"{v*100:+.1f}%",
        )
        combined = " ".join(bullets)
        assert "visibility" in combined or "Arena" in combined

    # _build_actions_block bucket reallocation ─────────────────────────────────

    def test_actions_bucket_reallocation_scale_and_pause(self):
        """Scale + Pause+Diagnose brand groups → 'Reallocation by bucket' line."""
        from report.weekly_brief import _build_actions_block
        from features.regimes import RegimeSignal
        from features.asin_metrics import ProductGroupMetrics
        sig = RegimeSignal(regime="baseline", active=True, confidence="High",
                           verdict="No regime", driver_type="Unknown")
        g_scale = ProductGroupMetrics("serum", "MyBrand", 2, 0.30, 0.0, 0.1,
                                      "gaining", "Scale", [])
        g_pause = ProductGroupMetrics("cleanser", "MyBrand", 2, 0.10, 0.1, 0.5,
                                      "losing", "Pause+Diagnose", [])
        metrics = {"A01": _make_metrics("A01", "MyBrand", ads_stance="Scale")}
        result = _build_actions_block(
            [sig], metrics, "MyBrand", "Hold", None,
            group_metrics=[g_scale, g_pause],
        )
        realloc = result[1]
        assert "Reallocation by bucket" in realloc
        assert "Serum" in realloc or "Scale" in realloc
        assert "Cleanser" in realloc or "Pause" in realloc


# ─── 10. Round 8: concerns taxonomy + nested arenas ──────────────────────────

class TestRound8:
    """Round 8: concerns taxonomy, concern metrics, pressure buckets, opportunity bucket,
    grouped receipts, cross-segment actions."""

    def test_tag_concerns_niacinamide_vitamin_c(self):
        from config.market_misattribution_module import tag_concerns
        result = tag_concerns("Niacinamide 10% + Vitamin C 20% Serum")
        assert "niacinamide" in result
        assert "vitamin_c" in result
        assert len(result) <= 2

    def test_tag_concerns_retinol_only(self):
        from config.market_misattribution_module import tag_concerns
        assert tag_concerns("Retinol 0.3% Serum for Face") == ["retinol"]

    def test_tag_concerns_no_match_returns_empty(self):
        from config.market_misattribution_module import tag_concerns
        assert tag_concerns("Gentle Foaming Cleanser") == []

    def test_tag_concerns_max_2(self):
        from config.market_misattribution_module import tag_concerns
        result = tag_concerns("Retinol Niacinamide Vitamin C Glycolic Serum")
        assert len(result) == 2

    def test_asin_metrics_has_concerns_field(self):
        m = _make_metrics("A01", "Brand")
        assert hasattr(m, "concerns")
        assert isinstance(m.concerns, list)

    def test_product_group_has_pct_losing(self):
        from features.asin_metrics import ProductGroupMetrics
        g = ProductGroupMetrics("serum", "Brand", 2, 0.1, 0.0, 0.2, "losing", "Pause+Diagnose", [], 0.6)
        assert g.pct_losing == 0.6

    def test_product_group_pct_losing_defaults_to_zero(self):
        from features.asin_metrics import ProductGroupMetrics
        g = ProductGroupMetrics("serum", "Brand", 2, 0.1, 0.0, 0.2, "gaining", "Scale", [])
        assert g.pct_losing == 0.0

    def test_compute_concern_metrics_groups_by_concern(self):
        from features.asin_metrics import compute_concern_metrics
        metrics = {
            "A01": _make_metrics("A01", "B1", concerns=["retinol", "niacinamide"]),
            "A02": _make_metrics("A02", "B2", concerns=["retinol"]),
        }
        df = _make_df_weekly([("A01", "B1", 25.0, 2000, 2100),
                              ("A02", "B2", 25.0, 3000, 3100)])
        result = compute_concern_metrics(metrics, df, "B1")
        concerns = {g.concern for g in result}
        assert "retinol" in concerns
        assert "niacinamide" in concerns

    def test_pressure_score_high_when_losing_and_discounted(self):
        from features.asin_metrics import _pressure_score
        score = _pressure_score(pct_losing=0.8, pct_discounted=0.7, median_price_vs_tier=0.15)
        assert score > 0.6

    def test_pressure_score_low_when_gaining(self):
        from features.asin_metrics import _pressure_score
        score = _pressure_score(pct_losing=0.0, pct_discounted=0.1, median_price_vs_tier=-0.05)
        assert score < 0.1

    def test_build_pressure_buckets_returns_top_2(self):
        from report.weekly_brief import _build_pressure_buckets
        from features.asin_metrics import ProductGroupMetrics
        groups = [
            ProductGroupMetrics("cleanser", "B1", 3, 0.2, 0.10, 0.6, "losing", "Pause+Diagnose", [], 0.7),
            ProductGroupMetrics("cleanser", "B2", 2, 0.15, 0.05, 0.5, "losing", "Pause+Diagnose", [], 0.6),
            ProductGroupMetrics("serum", "B1", 2, 0.3, -0.05, 0.2, "gaining", "Scale", [], 0.1),
            ProductGroupMetrics("serum", "B2", 3, 0.25, -0.10, 0.3, "mixed", "Hold", [], 0.2),
            ProductGroupMetrics("moisturizer", "B1", 2, 0.1, 0.15, 0.5, "losing", "Pause+Diagnose", [], 0.5),
        ]
        result = _build_pressure_buckets(groups, {}, "B1")
        assert len(result) <= 2
        assert all(hasattr(d, "claim") and hasattr(d, "receipts") for d in result)

    def test_grouped_receipts_list_shows_bucket_heading(self):
        from report.weekly_brief import grouped_receipts_list
        metrics = {
            "A01": _make_metrics("A01", "MyBrand", product_type="cleanser",
                                 ads_stance="Pause+Diagnose", bsr_wow=0.15),
        }
        result = grouped_receipts_list(
            metrics, "MyBrand", pressure_ptypes=["cleanser"],
        )
        combined = "\n".join(result)
        assert "Cleanser" in combined


class TestPhaseA:
    """Phase A: salesRankDrops, returnRate, activeIngredients, monthlySoldDelta, topCompBBShare."""

    def test_tag_concerns_ingredients_supplements_title(self):
        from config.market_misattribution_module import tag_concerns
        # Title doesn't mention niacinamide, ingredients string does
        result = tag_concerns("Daily Face Moisturizer 50ml", ingredients="Niacinamide 5%")
        assert "niacinamide" in result

    def test_tag_concerns_title_still_primary(self):
        from config.market_misattribution_module import tag_concerns
        result = tag_concerns("Retinol 0.3% Night Cream", ingredients="Shea Butter")
        assert result[0] == "retinol"

    def test_tag_concerns_empty_ingredients_unchanged(self):
        from config.market_misattribution_module import tag_concerns
        assert tag_concerns("Gentle Cleanser") == tag_concerns("Gentle Cleanser", ingredients="")

    def test_asin_metrics_has_return_rate_field(self):
        m = _make_metrics("A01", "Brand", return_rate=2)
        assert m.return_rate == 2

    def test_asin_metrics_has_sales_rank_drops(self):
        m = _make_metrics("A01", "Brand", sales_rank_drops_30=45)
        assert m.sales_rank_drops_30 == 45

    def test_return_rate_2_forces_high_ad_waste_risk(self):
        """High-return ASIN should get High ad_waste_risk even if price is competitive."""
        from features.asin_metrics import compute_asin_metrics
        df = _make_df_weekly([("A01", "MyBrand", 20.0, 2000, 2100),
                              ("A02", "Comp", 20.0, 3000, 3100)])
        df.loc[df["asin"] == "A01", "return_rate"] = 2
        metrics = compute_asin_metrics(df, {}, {}, "MyBrand")
        assert metrics["A01"].ad_waste_risk == "High"

    def test_return_rate_none_does_not_change_risk(self):
        from features.asin_metrics import compute_asin_metrics
        df = _make_df_weekly([("A01", "MyBrand", 18.0, 500, 480),
                              ("A02", "Comp", 20.0, 3000, 3100)])
        metrics = compute_asin_metrics(df, {}, {}, "MyBrand")
        # Good standing ASIN with no return_rate should not be forced to High
        assert metrics["A01"].ad_waste_risk in ("Low", "Med")

    def test_top_comp_bb_share_30_field_exists(self):
        m = _make_metrics("A01", "Brand", top_comp_bb_share_30=0.35)
        assert m.top_comp_bb_share_30 == 0.35

    def test_top_comp_bb_share_none_when_missing(self):
        """Locks semantic: None means no data, not zero."""
        m = _make_metrics("A01", "Brand", top_comp_bb_share_30=None)
        assert m.top_comp_bb_share_30 is None

    def test_ad_waste_reason_populated_for_high_return_rate(self):
        """ad_waste_reason should explain why ad_waste_risk is High."""
        from features.asin_metrics import compute_asin_metrics
        df = _make_df_weekly([("A01", "MyBrand", 20.0, 2000, 2100),
                              ("A02", "Comp", 20.0, 3000, 3100)])
        df.loc[df["asin"] == "A01", "return_rate"] = 2
        metrics = compute_asin_metrics(df, {}, {}, "MyBrand")
        assert metrics["A01"].ad_waste_reason is not None
        assert "return" in metrics["A01"].ad_waste_reason.lower()

    def test_item_type_keyword_fallback_for_product_type(self):
        """item_type_keyword should resolve 'other' when title parsing fails."""
        from config.market_misattribution_module import classify_title
        # Title alone → "other"
        assert classify_title("Amazing Product 50ml") == "other"
        # With item_type_keyword → correct type
        assert classify_title("Amazing Product 50ml", item_type_keyword="facial_serum") == "serum"
        assert classify_title("Amazing Product 50ml", item_type_keyword="facial_moisturizer") == "moisturizer"
        # Title match still takes priority
        assert classify_title("Retinol Night Serum 30ml", item_type_keyword="facial_moisturizer") == "retinol"

    def test_phase_a_receipt_extras_demand_signal(self):
        """phase_a_receipt_extras should surface demand delta when present and has_monthly_sold_history=True."""
        from features.asin_metrics import phase_a_receipt_extras
        m = _make_metrics("A01", "Brand", monthly_sold_delta=150, has_monthly_sold_history=True)
        result = phase_a_receipt_extras(m)
        assert "demand +150" in result

    def test_phase_a_receipt_extras_bb_unavailable_when_no_stats(self):
        """phase_a_receipt_extras should show 'Buy Box signal unavailable' when has_buybox_stats=False."""
        from features.asin_metrics import phase_a_receipt_extras
        m = _make_metrics("A01", "Brand")
        result = phase_a_receipt_extras(m)
        assert "Buy Box signal unavailable" in result

    def test_phase_a_receipt_extras_clean_when_all_flags_true_no_signals(self):
        """phase_a_receipt_extras should return empty string when flags are true but no signal values."""
        from features.asin_metrics import phase_a_receipt_extras
        m = _make_metrics("A01", "Brand", has_buybox_stats=True, has_monthly_sold_history=True)
        assert phase_a_receipt_extras(m) == ""