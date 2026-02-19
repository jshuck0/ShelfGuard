# 🔭 ShelfGuard

> Keepa-powered Amazon market intelligence that turns marketplace signals into weekly category briefs.

**Seller Central shows your store. ShelfGuard shows your market.**

Seller Central is siloed around your storefront and operations — it doesn't provide a clean view of category dynamics or competitor behavior. ShelfGuard builds a competitive set from a seed ASIN, computes brand-vs-market signals (pricing, promos, visibility/BSR), and delivers a weekly category readout plus the SKUs that need attention.

That context is especially valuable for marketers, because attribution is hard without knowing what the market was doing around you. ShelfGuard helps teams interpret swings correctly — separating category-wide pressure from brand-specific changes and highlighting the SKUs that actually drove the week. With a shared category read, marketing and ecommerce align faster on posture (hold, defend, investigate) without debating the story.

---

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white)
![Keepa](https://img.shields.io/badge/Keepa-API-orange)

---

## What It Does

Given a seed product and a brand name, ShelfGuard:

1. **Maps a competitive set** — scans Keepa category leaves to find the top ASINs competing in the same space (200–500 SKUs)
2. **Pulls 90 days of history** — price, BSR, promo activity, estimated sales for every ASIN in the set
3. **Computes brand-vs-market signals** — weekly changes for your brand benchmarked against the full arena
4. **Generates a weekly brief** — market environment verdict, leaf-level signals, brand SKU receipts, and a recommended posture

Weekly change is computed as last 7 days vs prior 7. 30–90 day history is used for multi-week context: to establish a baseline, detect sustained trends, and flag volatility so you don't overreact to one-week noise.

All signals are Keepa marketplace data only — no Seller Central, no ad spend, no CVR.

---

## The Brief

Each run produces a structured markdown brief with seven sections:

| Section | What It Answers |
|---------|-----------------|
| **Market Environment** | What regime is the category in this week? (Baseline / Promo Pressure / Price Pressure / Disruption / Rotation) |
| **Leaf Signals** | Which product-type buckets are under pressure or gaining ground? |
| **Active Signals** | Any ingredient or concern-level movements worth watching? |
| **Key SKUs** | Which brand SKUs drove the week — and what should you do about each? |
| **Plan** | Recommended posture: stance, budget direction, focus areas |
| **What to Watch** | Leading indicators to track next week |
| **Recommended Set** | Prior-week call accuracy (scoreboard) |

Brief title uses the dominant leaf category from the scanned set (e.g. _Naturium Brief — Face Serum_). Secondary leaves ≥10% of the set are disclosed as a subheader.

---

## Signals Computed

| Signal | Description |
|--------|-------------|
| **Visibility WoW** | BSR change week-over-week, sign-corrected (positive = gaining visibility) |
| **Price vs median** | Brand price positioned above / in line / below category median |
| **Promo activity** | Discount persistence: Low (0–1 days/wk) · Medium (2–4) · High (5–7) |
| **Est. revenue share** | Marketplace-observable proxy from BSR + price, not actual sales |
| **Sales rank drops** | 30- and 90-day drop counts from Keepa (volatility signal) |
| **Return rate** | Keepa return rate flag — gates ad-waste risk |
| **Demand delta** | Month-over-month units sold change (where Keepa history is available) |
| **Competitor BB share** | Top non-Amazon Buy Box holder's 30-day share |

---

## Architecture

```
ShelfGuard/
├── apps/
│   ├── mvp_app.py              # Main Streamlit entrypoint
│   └── search_to_state_ui.py  # Discovery UI components
│
├── src/
│   └── two_phase_discovery.py  # Competitive set mapping (Phase 1 + 2)
│
├── scrapers/
│   └── keepa_client.py         # Keepa API ingestion
│
├── features/
│   ├── asin_metrics.py         # Per-ASIN metrics + group analytics
│   └── regimes.py              # 5 market regime detectors
│
├── report/
│   └── weekly_brief.py         # Brief assembly + markdown renderer + Streamlit tab
│
├── config/
│   ├── market_misattribution_module.py  # Thresholds, taxonomy, band logic
│   └── golden_run.py                    # Pre-configured seed brand
│
├── scoring/
│   └── confidence.py           # HIGH / MED / LOW confidence rubric
│
└── tests/
    └── test_brief_logic.py     # 127 deterministic tests
```

The brief engine is fully rule-based — no LLM involved. All section logic is deterministic, threshold-driven, and covered by tests.

---

## Quick Start

```bash
git clone https://github.com/jshuck0/ShelfGuard.git
cd ShelfGuard
pip install -r requirements.txt

# Add keys to .streamlit/secrets.toml:
#   keepa.api_key = "..."
#   supabase.url  = "..."
#   supabase.key  = "..."

streamlit run apps/mvp_app.py
```

### Two modes

**Manual** — enter a keyword, pick a seed product, name your brand, click Map Market.

**Golden run** — configure `config/golden_run.py` with a seed ASIN and brand, then click Load Market for a one-click pre-configured run.

---

## Built With

- **Keepa API** — market data (price, BSR, promo, sales history)
- **Streamlit** — UI
- **Supabase** — optional caching for instant return visits
- **Pandas / NumPy** — signal computation
