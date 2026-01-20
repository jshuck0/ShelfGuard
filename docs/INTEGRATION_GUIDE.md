# ShelfGuard Strategic Intelligence System - Integration Guide

**Version:** 1.0
**Date:** 2026-01-19
**Status:** Implementation Complete - Ready for Integration

---

## Overview

This guide shows how to integrate the new Strategic Intelligence System with existing ShelfGuard components, specifically Phase 2 Discovery and the Command Center UI.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2 DISCOVERY (Existing)                 │
│  • Keepa Product Finder (keyword/category search)              │
│  • 90-day historical data fetch                                  │
│  • Synthetic intelligence (financial gap filling)               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│          NEW: INTELLIGENCE PIPELINE (This System)               │
│                                                                  │
│  ┌─────────────────┐      ┌──────────────────┐                │
│  │  1. Triggers    │──────▶│  2. Network      │                │
│  │     Detection   │       │     Intelligence │                │
│  └─────────────────┘       └──────────────────┘                │
│           │                         │                            │
│           ▼                         ▼                            │
│  ┌─────────────────┐      ┌──────────────────┐                │
│  │  3. AI Engine   │──────▶│  4. AI Engine    │                │
│  │     v2 (Class.) │       │     v3 (Insight) │                │
│  └─────────────────┘       └──────────────────┘                │
│           │                         │                            │
│           └────────┬────────────────┘                           │
│                    ▼                                             │
│           ┌────────────────┐                                    │
│           │ 5. Unified     │                                    │
│           │    Intelligence│                                    │
│           └────────────────┘                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SUPABASE DATABASE                               │
│  • strategic_insights                                            │
│  • trigger_events                                                │
│  • category_intelligence (accumulates over time)                │
│  • brand_intelligence                                            │
│  • market_patterns                                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              COMMAND CENTER UI (Updated)                         │
│  • Action Queue (filtered by priority)                          │
│  • Network Intelligence displays                                │
│  • Trigger event visualizations                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Steps

### Step 1: Hook Intelligence Pipeline into Phase 2 Discovery

Add this code to `src/two_phase_discovery.py` after Phase 2 completion:

```python
from src.intelligence_pipeline import IntelligencePipeline
from supabase import create_client
import os

def phase2_market_mapping_with_intelligence(
    seed_asin: str,
    category_id: int,
    limit: int = 100
) -> Tuple[pd.DataFrame, List[UnifiedIntelligence]]:
    """
    Phase 2: Market mapping with intelligence generation.

    Returns:
        Tuple of (market_snapshot_df, intelligence_results)
    """

    # EXISTING PHASE 2 CODE (unchanged)
    # 1. Product Finder to get ASINs in category
    asins = product_finder_by_category(category_id, limit=limit)

    # 2. Fetch 90-day history for all ASINs
    historical_data = fetch_90day_history_batch(asins)

    # 3. Build market snapshot DataFrame
    market_snapshot = build_market_snapshot(historical_data)

    # 4. Apply synthetic intelligence (financial enrichment)
    from apps.synthetic_intel import enrich_synthetic_financials
    market_snapshot = enrich_synthetic_financials(market_snapshot)

    # NEW: INTELLIGENCE PIPELINE INTEGRATION
    # Initialize pipeline
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )

    pipeline = IntelligencePipeline(
        supabase=supabase,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        enable_data_accumulation=True  # Enable network effect
    )

    # Accumulate market data (runs in background)
    category_context = {
        'category_id': category_id,
        'category_name': 'Extracted from seed',  # Extract from seed ASIN
        'category_tree': []  # Extract breadcrumb
    }

    pipeline.accumulate_market_data(
        market_snapshot=market_snapshot,
        category_id=category_id,
        category_name=category_context['category_name'],
        category_tree=category_context['category_tree']
    )

    # Generate intelligence for portfolio ASINs
    # (Only analyze user's tracked ASINs, not entire market)
    portfolio_asins = get_user_tracked_asins()  # Your existing function

    # Prepare market data format
    market_data = prepare_market_data_for_pipeline(
        portfolio_asins,
        historical_data,
        market_snapshot
    )

    # Generate unified intelligence
    intelligence_results = pipeline.generate_portfolio_intelligence(
        portfolio_asins=portfolio_asins,
        market_data=market_data,
        category_context=category_context
    )

    return market_snapshot, intelligence_results


def prepare_market_data_for_pipeline(
    portfolio_asins: List[str],
    historical_data: Dict[str, pd.DataFrame],
    market_snapshot: pd.DataFrame
) -> Dict[str, Any]:
    """
    Transform Phase 2 data into format expected by intelligence pipeline.
    """
    market_data = {}

    for asin in portfolio_asins:
        # Get competitor data (other ASINs in same category)
        competitor_data = market_snapshot[market_snapshot['asin'] != asin].copy()

        # Get current metrics
        current_row = market_snapshot[market_snapshot['asin'] == asin]
        if current_row.empty:
            continue

        current_metrics = {
            'price': current_row['filled_price'].iloc[0],
            'bsr': current_row['sales_rank_filled'].iloc[0],
            'review_count': current_row.get('review_count', {}).iloc[0],
            'rating': current_row.get('rating', 0).iloc[0],
            'buybox_share': current_row.get('buybox_share', 1.0).iloc[0],
            'inventory': current_row.get('inventory_count', 0).iloc[0],
            'brand': current_row.get('brand', 'Unknown').iloc[0],
            'category_root': current_row.get('category_root', '').iloc[0],
            'estimated_monthly_sales': current_row.get('est_monthly_sales', 0).iloc[0],
            'estimated_monthly_revenue': current_row.get('est_monthly_revenue', 0).iloc[0]
        }

        market_data[asin] = {
            'historical': historical_data.get(asin, pd.DataFrame()),
            'competitors': competitor_data,
            'current_metrics': current_metrics
        }

    return market_data
```

---

### Step 2: Update Command Center UI

Update `apps/shelfguard_app.py` or your main UI file:

```python
from src.intelligence_pipeline import get_active_insights_from_db
from src.models.product_status import ProductStatus

def render_action_queue():
    """
    Render Action Queue with new unified intelligence.
    """
    st.header("⚡ Action Queue")

    # Get active insights from database
    supabase = get_supabase_client()
    insights = get_active_insights_from_db(
        supabase=supabase,
        user_id=st.session_state.get('user_id'),
        priority_threshold=50  # Only show WATCH+ priority
    )

    if not insights:
        st.info("No active insights. Run Phase 2 Discovery to generate intelligence.")
        return

    # Group by priority tier
    critical = [i for i in insights if ProductStatus[i['product_status'].upper()].priority == 100]
    opportunities = [i for i in insights if ProductStatus[i['product_status'].upper()].priority == 75]
    watch = [i for i in insights if ProductStatus[i['product_status'].upper()].priority == 50]

    # Render CRITICAL section (collapsed by default if empty)
    with st.expander(f"🚨 CRITICAL ({len(critical)})", expanded=len(critical) > 0):
        for insight in critical:
            render_insight_card(insight, priority="critical")

    # Render OPPORTUNITY section
    with st.expander(f"💡 OPPORTUNITIES ({len(opportunities)})", expanded=len(opportunities) > 0):
        for insight in opportunities:
            render_insight_card(insight, priority="opportunity")

    # Render WATCH section (collapsed by default)
    with st.expander(f"👀 WATCH ({len(watch)})", expanded=False):
        for insight in watch:
            render_insight_card(insight, priority="watch")


def render_insight_card(insight: Dict[str, Any], priority: str):
    """
    Render a single insight card with trigger events and network context.
    """
    asin = insight['asin']

    # Card container
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{asin}** - {insight['strategic_state']}")
            st.caption(insight['recommendation'])

        with col2:
            st.metric(
                "Upside",
                f"${insight['projected_upside_monthly']:.0f}/mo",
                delta=f"+{insight['net_expected_value']:.0f}",
                delta_color="normal"
            )

        with col3:
            st.metric(
                "Risk",
                f"${insight['downside_risk_monthly']:.0f}/mo",
                delta=f"{insight['confidence']:.0f}% confidence"
            )

        # Show trigger events
        if st.button(f"View Details", key=f"details_{insight['id']}"):
            show_insight_details(insight)

        st.divider()


def show_insight_details(insight: Dict[str, Any]):
    """
    Show full insight details including trigger events and reasoning.
    """
    st.subheader(f"Insight Details: {insight['asin']}")

    # Reasoning
    st.markdown("**AI Reasoning:**")
    st.info(insight['reasoning'])

    # Trigger Events
    st.markdown("**Trigger Events:**")
    supabase = get_supabase_client()
    triggers = supabase.table('trigger_events').select('*').eq(
        'insight_id', insight['id']
    ).order('severity', desc=True).execute()

    if triggers.data:
        for trigger in triggers.data:
            severity_emoji = "🔴" if trigger['severity'] >= 8 else "🟡" if trigger['severity'] >= 6 else "🟢"
            st.markdown(
                f"{severity_emoji} **{trigger['event_type']}** (Severity: {trigger['severity']}/10)  \n"
                f"   {trigger['metric_name']}: {trigger['baseline_value']:.2f} → {trigger['current_value']:.2f} "
                f"({trigger['delta_pct']:+.1f}%)"
            )
    else:
        st.caption("No trigger events for this insight.")

    # Action plan
    st.markdown("**Recommended Action:**")
    st.success(f"{insight['action_type'].upper()}: {insight['recommendation']}")
    st.caption(f"Time Horizon: {insight['time_horizon_days']} days")
```

---

### Step 3: Add Network Intelligence Dashboard

Create a new section showing network effect in action:

```python
def render_network_intelligence_dashboard():
    """
    Show accumulated network intelligence (category benchmarks, brand data).
    """
    st.header("🌐 Network Intelligence")

    supabase = get_supabase_client()

    # Show category intelligence
    st.subheader("Category Benchmarks")

    category_intel = supabase.table('category_intelligence').select('*').order(
        'snapshot_date', desc=True
    ).limit(10).execute()

    if category_intel.data:
        df_categories = pd.DataFrame(category_intel.data)

        st.dataframe(
            df_categories[[
                'category_id', 'category_name', 'median_price',
                'median_review_count', 'total_asins_tracked', 'data_quality'
            ]],
            use_container_width=True
        )

        # Visualization: Price trends by category
        st.line_chart(df_categories.set_index('snapshot_date')['median_price'])
    else:
        st.info("No category benchmarks yet. Data accumulates as you search products.")

    # Show brand intelligence
    st.subheader("Brand Intelligence")

    brand_intel = supabase.table('brand_intelligence').select('*').order(
        'last_updated', desc=True
    ).limit(10).execute()

    if brand_intel.data:
        df_brands = pd.DataFrame(brand_intel.data)
        st.dataframe(
            df_brands[[
                'brand', 'category_root', 'avg_price', 'avg_review_count',
                'product_count', 'market_share_pct'
            ]],
            use_container_width=True
        )
    else:
        st.info("No brand intelligence yet. Data accumulates as you search products.")

    # Show market patterns
    st.subheader("Discovered Patterns")

    patterns = supabase.table('market_patterns').select('*').order(
        'confidence', desc=True
    ).limit(5).execute()

    if patterns.data:
        for pattern in patterns.data:
            st.markdown(
                f"**{pattern['pattern_type']}**  \n"
                f"Typical Outcome: {pattern['typical_outcome']}  \n"
                f"Confidence: {pattern['confidence']:.1%} (Observed {pattern['observed_count']} times)"
            )
            st.divider()
    else:
        st.info("No patterns discovered yet. System learns patterns over time.")
```

---

## Data Flow Summary

### On Every Phase 2 Discovery Run:

1. **Phase 2 Discovery runs** (existing code, unchanged)
   - Keepa Product Finder searches category
   - Fetches 90-day historical data
   - Builds market snapshot DataFrame
   - Applies synthetic financial enrichment

2. **Network Intelligence Accumulator runs** (new, automatic)
   - Stores all product snapshots in `product_snapshots` table
   - Calculates category benchmarks → `category_intelligence`
   - Updates brand aggregates → `brand_intelligence`
   - Detects patterns → `market_patterns`

3. **Intelligence Pipeline runs** (new, for portfolio ASINs only)
   - Detects trigger events (inventory, price wars, etc.)
   - Queries accumulated network intelligence
   - Calls AI Engine v2 for classification
   - Calls AI Engine v3 for insights
   - Stores results in `strategic_insights` + `trigger_events`

4. **UI displays results** (updated)
   - Action Queue shows prioritized insights
   - Network Intelligence dashboard shows accumulated data
   - Trigger events visible in detail views

---

## Compatibility Notes

### Synthetic Intelligence Module

The existing `apps/synthetic_intel.py` module is **100% compatible** with the new architecture:

- **What it does:** Enriches DataFrames with synthetic financial calculations (COGS, volumetric data, logistics costs)
- **Where it runs:** After Phase 2 builds market snapshot, before intelligence pipeline
- **Data flow:** `market_snapshot` → `enrich_synthetic_financials()` → `enriched_snapshot` → `intelligence_pipeline`

**No changes required** to synthetic intelligence module. It continues to work as-is.

### Supabase Caching

The new intelligence pipeline stores results in Supabase for:
- Historical tracking (outcome validation)
- Cross-session persistence (Action Queue)
- Network effect (accumulated benchmarks)

**Caching strategy:**
- `@st.cache_data(ttl=3600)` for Phase 2 discovery (existing, unchanged)
- No caching for intelligence pipeline (always runs fresh to detect latest triggers)
- Supabase acts as persistent cache for accumulated data

---

## Environment Variables Required

Add these to your `.env` file:

```bash
# OpenAI (required for AI engines v2/v3)
OPENAI_API_KEY=sk-...

# Supabase (required for database storage)
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# Keepa (existing, required for Phase 2)
KEEPA_API_KEY=your_keepa_key
```

---

## Testing the Integration

### Quick Test:

1. Run Phase 2 Discovery on a category
2. Check Supabase tables populated:
   ```sql
   SELECT COUNT(*) FROM product_snapshots;
   SELECT COUNT(*) FROM category_intelligence;
   SELECT COUNT(*) FROM strategic_insights;
   ```
3. View Action Queue in UI - should show insights
4. Check Network Intelligence dashboard - should show accumulated data

### Full Test Scenario:

```python
# In your test script or notebook
from src.intelligence_pipeline import IntelligencePipeline
from supabase import create_client
import os

# Initialize
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

pipeline = IntelligencePipeline(
    supabase=supabase,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Test single ASIN
test_market_data = {
    'B01234ASIN': {
        'historical': historical_df,  # Your 90-day data
        'competitors': competitors_df,
        'current_metrics': {
            'price': 24.99,
            'bsr': 5000,
            # ...
        }
    }
}

results = pipeline.generate_portfolio_intelligence(
    portfolio_asins=['B01234ASIN'],
    market_data=test_market_data,
    category_context={'category_id': 16310101, ...}
)

# Verify result
print(results[0].recommendation)
print(f"Upside: ${results[0].projected_upside_monthly}/mo")
print(f"Triggers: {len(results[0].trigger_events)}")
```

---

## Performance Considerations

### Expected Latency:

- **Phase 2 Discovery**: 30-60 seconds (unchanged, Keepa API bound)
- **Network Accumulation**: 5-10 seconds (runs in parallel, database writes)
- **Intelligence Pipeline**:
  - Trigger detection: <1 second per ASIN
  - Network queries: <1 second per ASIN
  - AI Classification (v2): 2-3 seconds per ASIN (LLM call)
  - AI Insight (v3): 2-3 seconds per ASIN (LLM call)
  - **Total**: ~5-7 seconds per ASIN

For portfolio of 10 ASINs: ~60-70 seconds total intelligence generation.

### Optimization Tips:

1. **Run intelligence pipeline in background:**
   ```python
   import threading

   def async_generate_intelligence():
       pipeline.generate_portfolio_intelligence(...)

   thread = threading.Thread(target=async_generate_intelligence)
   thread.start()

   # Show "Generating insights..." spinner in UI
   # Load from database once complete
   ```

2. **Batch LLM calls:** Future optimization to call OpenAI batch API for multiple ASINs

3. **Cache network intelligence:** Category benchmarks change slowly, can cache for 1 hour

---

## Migration Path

If you have existing ShelfGuard data/code:

1. **Week 1:** Run database migrations (DONE - you ran these already)
2. **Week 2:** Add intelligence pipeline hook to Phase 2 (code above)
3. **Week 3:** Update Action Queue UI (code above)
4. **Week 4:** Add Network Intelligence dashboard (optional, code above)
5. **Week 5:** Test with real portfolio, tune LLM prompts if needed

**No breaking changes** to existing Phase 2 Discovery or synthetic intelligence code.

---

## Support

Questions? Check:
- [MASTER_ARCHITECTURE_OVERVIEW.md](MASTER_ARCHITECTURE_OVERVIEW.md) - Full system design
- [IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md) - Current status
- [AI_PREDICTIVE_ENGINE_ARCHITECTURE.md](AI_PREDICTIVE_ENGINE_ARCHITECTURE.md) - LLM details
- [NETWORK_INTELLIGENCE_ARCHITECTURE.md](NETWORK_INTELLIGENCE_ARCHITECTURE.md) - Network effect details

---

**Last Updated:** 2026-01-19
**Status:** ✅ Ready for Integration
