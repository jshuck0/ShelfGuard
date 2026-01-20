# ShelfGuard Strategic Intelligence System
## Master Architecture Overview & Integration

**Status:** Complete Unified Architecture
**Author:** Technical Architecture Team
**Date:** 2026-01-19
**Version:** 2.0 - Network-Aware Intelligence

---

## Executive Summary

This document provides a consolidated view of ShelfGuard's complete Strategic Intelligence System architecture, integrating three major subsystems:

1. **Insight Engine** - High-fidelity recommendation generation
2. **AI & Predictive Intelligence** - LLM-powered classification and forecasting
3. **Network Intelligence** - Self-learning system via data accumulation

**Key Innovation:** These three systems work together in a unified pipeline where:
- Every user search enriches the network intelligence
- Network intelligence improves AI predictions
- AI predictions generate high-fidelity insights
- Insights drive user actions that generate more data

---

## Part 1: System Architecture Overview

### 1.1 The Complete Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         USER ACTION: Search Product                           │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: MARKET DISCOVERY                                │
│  - Fetch products from Keepa                                                 │
│  - Extract brand, price, BSR, reviews, inventory                             │
│  - Output: market_snapshot DataFrame                                         │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              DATA ACCUMULATION (NEW - Network Intelligence)                   │
│  1. Store snapshots in product_snapshots table                               │
│  2. Update category_intelligence (benchmarks)                                │
│  3. Update brand_intelligence (aggregates)                                   │
│  4. Detect and store market_patterns                                         │
│  Output: Network intelligence ready for querying                             │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TRIGGER EVENT DETECTION (Insight Engine)                   │
│  Scan historical data for:                                                   │
│  - Competitor inventory drops (<5 units)                                     │
│  - Price wars (3+ drops in 7d)                                               │
│  - Review velocity spikes                                                    │
│  - BuyBox share collapse                                                     │
│  - Rank degradation (30%+ worse)                                             │
│  Output: List[TriggerEvent] with severity rankings                           │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│          NETWORK INTELLIGENCE ENRICHMENT (Network Intelligence)               │
│  Query accumulated data:                                                     │
│  - Category benchmarks (median price, reviews, BSR from all users)          │
│  - Competitive position (percentile rankings)                                │
│  - Brand intelligence (brand-level aggregates)                               │
│  - Historical patterns (we've seen this before)                              │
│  - Similar products (for comparison)                                         │
│  Output: network_context Dict                                                │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│         STRATEGIC CLASSIFICATION (AI & Predictive Intelligence)               │
│  LLM Call #1: Classify product status                                        │
│  Input: Metrics + Triggers + Network Context                                │
│  Output: ProductStatus (13 granular statuses)                                │
│          StrategicState (5 strategic states)                                 │
│          Confidence score (boosted by network data quality)                  │
│          Reasoning (must cite triggers + network intelligence)               │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│      PREDICTIVE FORECASTING (AI & Predictive Intelligence)                   │
│  Calculate 30-day forecasts:                                                │
│  - Risk forecast (trigger-amplified)                                         │
│  - Growth forecast (opportunity detection)                                   │
│  - Stockout risk                                                             │
│  - Price erosion risk                                                        │
│  - Share erosion risk                                                        │
│  Output: PredictiveAlpha with quantified $ amounts                           │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│            INSIGHT GENERATION (Insight Engine)                                │
│  LLM Call #2: Generate specific recommendation                               │
│  Input: Status + Triggers + Predictions + Network Context                   │
│  Output: Recommendation with:                                                │
│          - Specific $ amount or %                                            │
│          - Causal reasoning citing triggers                                  │
│          - Projected upside/downside                                         │
│          - Time horizon                                                      │
│          - Action type (PROFIT_CAPTURE or RISK_MITIGATION)                   │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  QUALITY VALIDATION (Insight Engine)                          │
│  Validate insight meets requirements:                                        │
│  ✓ Has $ amount or percentage                                               │
│  ✓ Cites ≥1 trigger event                                                   │
│  ✓ Quantifies financial impact                                              │
│  ✓ Reasoning >50 characters                                                 │
│  ✓ Confidence 0.3-1.0                                                        │
│  Output: validation_passed (bool), validation_errors (list)                  │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                DATABASE STORAGE (Insight Engine)                              │
│  Store in strategic_insights table:                                          │
│  - Unified product_status                                                    │
│  - Recommendation                                                            │
│  - Trigger events (JSONB)                                                    │
│  - Financial impact (upside/downside)                                        │
│  - Action type                                                               │
│  - Validation status                                                         │
│  Also store in: trigger_events, insight_outcomes tables                      │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   ACTION QUEUE UI (Insight Engine)                            │
│  Display filtered insights:                                                  │
│  - Hide STABLE products by default                                           │
│  - Show top 25 by priority                                                   │
│  - Visual distinction: Green (opportunity) vs Red (risk)                     │
│  - Show trigger events, $ amounts, time horizon                              │
│  - Collect user feedback                                                     │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   FEEDBACK LOOP (Network Intelligence)                        │
│  Track outcomes:                                                             │
│  - Did user take action?                                                     │
│  - Did prediction come true?                                                 │
│  - Actual revenue impact vs predicted                                        │
│  Use to:                                                                     │
│  - Calibrate LLM confidence                                                  │
│  - Improve pattern recognition                                               │
│  - Refine trigger thresholds                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Three Subsystem Integration

### 2.1 Insight Engine (Layer 1: Output)

**Purpose:** Generate high-fidelity, actionable recommendations

**Key Components:**
- `ProductStatus` enum (13 granular statuses)
- `strategic_insights` table
- `trigger_events` table
- `insight_outcomes` table (for accuracy tracking)
- Quality validation gates
- Action Queue filtering logic

**Inputs from Other Systems:**
- Trigger events (self-generated)
- Strategic classification (from AI Engine)
- Predictive forecasts (from AI Engine)
- Network context (from Network Intelligence)

**Outputs:**
- Specific recommendations with $ amounts
- Filtered Action Queue (hide STABLE products)
- User-facing insights

**Document:** [INSIGHT_ENGINE_REFACTOR_PLAN.md](INSIGHT_ENGINE_REFACTOR_PLAN.md)

### 2.2 AI & Predictive Intelligence Engine (Layer 2: Analysis)

**Purpose:** Classify products and forecast 30-day outcomes

**Key Components:**
- `StrategicBrief` dataclass (existing)
- `UnifiedIntelligence` dataclass (new - combines strategic + predictive + insights)
- `PredictiveAlpha` dataclass (existing)
- Two-stage LLM architecture:
  - LLM Call #1: Strategic classification
  - LLM Call #2: Insight generation
- Trigger event amplification in risk forecasting

**Inputs from Other Systems:**
- Trigger events (from Insight Engine)
- Network context (from Network Intelligence)

**Outputs:**
- Strategic classification with confidence
- 30-day risk/growth forecasts
- Trigger-aware AI recommendations

**Document:** [AI_PREDICTIVE_ENGINE_ARCHITECTURE.md](AI_PREDICTIVE_ENGINE_ARCHITECTURE.md)

### 2.3 Network Intelligence Engine (Layer 3: Knowledge Base)

**Purpose:** Accumulate data and provide category/competitive context

**Key Components:**
- `product_snapshots` table (existing - central data warehouse)
- `category_intelligence` table (new - benchmarks)
- `brand_intelligence` table (new - brand aggregates)
- `market_patterns` table (new - historical patterns)
- `NetworkIntelligenceAccumulator` (writes data)
- `NetworkIntelligence` (reads data)

**Inputs from Other Systems:**
- Market snapshots from Phase 2 discovery
- User search activity

**Outputs:**
- Category benchmarks (median price, reviews, BSR)
- Competitive position analysis (percentile rankings)
- Brand intelligence (brand-level metrics)
- Historical patterns ("we've seen this before")
- Confidence boost based on data quality

**Document:** [NETWORK_INTELLIGENCE_ARCHITECTURE.md](NETWORK_INTELLIGENCE_ARCHITECTURE.md)

---

## Part 3: Database Schema Integration

### 3.1 Existing Tables (Leverage)

```sql
-- Already exists in search_to_state.sql
product_snapshots (
    asin, snapshot_date,
    buy_box_price, sales_rank, review_count, rating,
    estimated_weekly_revenue, estimated_units,
    title, brand, category, main_image
)
```

### 3.2 New Tables (Add)

**From Insight Engine:**
```sql
strategic_insights (
    asin, product_status, recommendation, reasoning,
    trigger_events (JSONB), projected_upside_monthly, downside_risk_monthly,
    action_type, time_horizon_days, validation_passed
)

trigger_events (
    asin, event_type, severity, metric_name,
    baseline_value, current_value, delta_pct,
    related_asin
)

insight_outcomes (
    insight_id, action_taken, actual_outcome,
    actual_revenue_impact, predicted_revenue_impact,
    prediction_error_pct
)
```

**From Network Intelligence:**
```sql
-- Extend product_snapshots
ALTER TABLE product_snapshots ADD COLUMN category_id INTEGER;
ALTER TABLE product_snapshots ADD COLUMN category_name TEXT;
ALTER TABLE product_snapshots ADD COLUMN category_tree TEXT[];
ALTER TABLE product_snapshots ADD COLUMN category_root TEXT;

-- New aggregate tables
category_intelligence (
    category_id, snapshot_date,
    median_price, p75_price, p25_price,
    median_rating, median_review_count, median_bsr,
    total_asins_tracked, data_quality
)

brand_intelligence (
    brand, category_root,
    total_asins_tracked, avg_price, avg_rating,
    total_weekly_revenue, market_share_pct
)

market_patterns (
    pattern_type, category_root,
    trigger_conditions (JSONB), typical_outcome,
    success_rate, avg_revenue_impact,
    observed_count, confidence_score
)
```

### 3.3 Schema Relationships

```
product_snapshots (CENTRAL WAREHOUSE)
    ├─→ category_intelligence (aggregated by category_id, snapshot_date)
    ├─→ brand_intelligence (aggregated by brand, category_root)
    └─→ market_patterns (patterns detected across products)

trigger_events
    └─→ strategic_insights (triggers linked to insights via trigger_events JSONB)

strategic_insights
    └─→ insight_outcomes (track prediction accuracy)
```

---

## Part 4: Code Architecture Integration

### 4.1 Main Entry Point

**File:** `src/intelligence_pipeline.py`

```python
async def generate_portfolio_intelligence(
    asins: List[str],
    lookback_days: int = 90,
    strategic_bias: str = "Balanced Defense"
) -> List[UnifiedIntelligence]:
    """
    UNIFIED PIPELINE - Orchestrates all three subsystems.

    Flow:
    1. Load historical data
    2. Detect trigger events (Insight Engine)
    3. Accumulate data to network (Network Intelligence)
    4. Query network context (Network Intelligence)
    5. Classify products with LLM (AI Engine)
    6. Generate predictions (AI Engine)
    7. Generate insights with LLM (Insight Engine)
    8. Validate quality (Insight Engine)
    9. Store to database
    10. Return unified intelligence objects
    """

    # Initialize clients
    supabase = get_supabase_client()
    accumulator = NetworkIntelligenceAccumulator(supabase)
    network_intel = NetworkIntelligence(supabase)
    openai_client = AsyncOpenAI()

    all_intelligence = []

    for asin in asins:
        # Load data
        row_data = get_current_metrics(asin)
        df_historical = load_historical_data(asin, lookback_days)
        df_competitors = load_competitor_data(asin)
        category_id = row_data['category_id']

        # ========== SUBSYSTEM 1: INSIGHT ENGINE ==========
        # Detect trigger events
        from src.trigger_detection import detect_trigger_events
        trigger_events = detect_trigger_events(
            asin, df_historical, df_competitors
        )

        # ========== SUBSYSTEM 2: NETWORK INTELLIGENCE ==========
        # Accumulate data (happens in background)
        accumulator.accumulate_search_data(
            market_snapshot=df_competitors,
            category_id=category_id,
            category_name=row_data['category_name'],
            category_tree=row_data['category_tree']
        )

        # Query network context
        category_benchmarks = network_intel.get_category_benchmarks(category_id)
        competitive_position = network_intel.get_competitive_position(asin, category_id)
        brand_intel = network_intel.get_brand_intelligence(
            row_data['brand'],
            category_benchmarks['category_root']
        )

        network_context = {
            'category_benchmarks': category_benchmarks,
            'competitive_position': competitive_position,
            'brand_intelligence': brand_intel
        }

        # ========== SUBSYSTEM 3: AI & PREDICTIVE ENGINE ==========
        # Classify product (LLM #1)
        from utils.ai_engine_v3 import analyze_product_with_network_intelligence

        intelligence = await analyze_product_with_network_intelligence(
            asin=asin,
            row_data=row_data,
            df_historical=df_historical,
            df_competitors=df_competitors,
            category_id=category_id,
            trigger_events=trigger_events,
            network_context=network_context,
            client=openai_client,
            strategic_bias=strategic_bias
        )

        # ========== SUBSYSTEM 1: INSIGHT ENGINE (Validation) ==========
        # Validate insight quality
        from src.insight_pipeline import validate_insight_quality
        validation_passed, validation_errors = validate_insight_quality(
            intelligence.to_dict()
        )

        intelligence.validation_passed = validation_passed
        intelligence.validation_errors = validation_errors

        # Store to database
        if validation_passed:
            await store_insight(intelligence, supabase)

        all_intelligence.append(intelligence)

    return all_intelligence
```

### 4.2 File Organization

```
ShelfGuard/
├── src/
│   ├── models/
│   │   ├── unified_intelligence.py      (NEW - UnifiedIntelligence dataclass)
│   │   ├── trigger_event.py             (NEW - TriggerEvent dataclass)
│   │   ├── product_status.py            (NEW - ProductStatus enum)
│   │   └── predictive_alpha.py          (EXISTING - PredictiveAlpha dataclass)
│   ├── trigger_detection.py             (NEW - Insight Engine)
│   ├── data_accumulation.py             (NEW - Network Intelligence)
│   ├── network_intelligence.py          (NEW - Network Intelligence queries)
│   ├── intelligence_pipeline.py         (NEW - Main orchestrator)
│   ├── insight_pipeline.py              (NEW - Insight Engine logic)
│   └── two_phase_discovery.py           (EXISTING - Phase 2 market discovery)
├── utils/
│   ├── ai_engine.py                     (EXISTING - Keep for backward compat)
│   ├── ai_engine_v2.py                  (NEW - Enhanced with triggers)
│   └── ai_engine_v3.py                  (NEW - Enhanced with network)
├── database/
│   └── insights_db.py                   (NEW - Database operations)
├── schemas/
│   ├── search_to_state.sql              (EXISTING - product_snapshots)
│   ├── strategic_insights.sql           (NEW - Insight Engine tables)
│   ├── network_intelligence.sql         (NEW - Network Intelligence tables)
│   └── migrations/
│       └── add_network_intelligence.sql (NEW - Schema extensions)
├── apps/
│   ├── shelfguard_app.py                (REFACTOR - Use new pipeline)
│   └── search_to_state_ui.py            (UPDATE - Hook accumulation)
└── docs/
    ├── INSIGHT_ENGINE_REFACTOR_PLAN.md
    ├── AI_PREDICTIVE_ENGINE_ARCHITECTURE.md
    ├── NETWORK_INTELLIGENCE_ARCHITECTURE.md
    └── MASTER_ARCHITECTURE_OVERVIEW.md  (THIS FILE)
```

---

## Part 5: Implementation Sequence

### Phase 1: Foundation (Week 1)

**Goal:** Set up database schema and data models

- [ ] Day 1-2: Create all new tables (strategic_insights, trigger_events, category_intelligence, brand_intelligence, market_patterns)
- [ ] Day 3: Extend product_snapshots with category columns
- [ ] Day 4: Create data models (UnifiedIntelligence, TriggerEvent, ProductStatus)
- [ ] Day 5: Set up database client in `database/insights_db.py`

### Phase 2: Trigger Detection (Week 2)

**Goal:** Build trigger event detection system

- [ ] Day 1-2: Implement `src/trigger_detection.py` (6 core detectors)
- [ ] Day 3: Unit tests for trigger detection
- [ ] Day 4: Integrate into existing Phase 2 discovery
- [ ] Day 5: Test trigger detection on real data

### Phase 3: Network Intelligence (Week 3)

**Goal:** Build data accumulation and query layer

- [ ] Day 1-2: Implement `src/data_accumulation.py` (NetworkIntelligenceAccumulator)
- [ ] Day 3: Implement `src/network_intelligence.py` (query layer)
- [ ] Day 4: Hook accumulation into Phase 2 completion
- [ ] Day 5: Test network intelligence queries

### Phase 4: AI Engine Enhancement (Week 4)

**Goal:** Enhance AI engine with triggers and network context

- [ ] Day 1-2: Create `utils/ai_engine_v2.py` (trigger-aware prompts)
- [ ] Day 3: Create `utils/ai_engine_v3.py` (network-aware prompts)
- [ ] Day 4: Implement two-stage LLM architecture
- [ ] Day 5: Test LLM outputs with quality validation

### Phase 5: Unified Pipeline (Week 5)

**Goal:** Build main orchestrator and integrate all systems

- [ ] Day 1-2: Create `src/intelligence_pipeline.py` (main entry point)
- [ ] Day 3: Implement quality validation gates
- [ ] Day 4: Implement database storage logic
- [ ] Day 5: End-to-end testing

### Phase 6: UI Integration (Week 6)

**Goal:** Update dashboard to use new pipeline

- [ ] Day 1-2: Update `apps/shelfguard_app.py` to use intelligence_pipeline
- [ ] Day 3: Update Action Queue rendering (filtered, styled)
- [ ] Day 4: Add network intelligence displays (benchmarks, position)
- [ ] Day 5: Add trigger event visualizations

### Phase 7: Testing & Optimization (Week 7)

**Goal:** Validate system, optimize performance

- [ ] Day 1: Unit tests for all components
- [ ] Day 2: Integration tests (end-to-end)
- [ ] Day 3: Load testing (100+ ASINs)
- [ ] Day 4: LLM prompt optimization
- [ ] Day 5: Performance tuning

### Phase 8: Launch & Monitoring (Week 8)

**Goal:** Deploy to production, monitor metrics

- [ ] Day 1: Production deployment (staged rollout)
- [ ] Day 2: Monitor network intelligence growth
- [ ] Day 3: Monitor LLM call success rates
- [ ] Day 4: Monitor insight quality metrics
- [ ] Day 5: Collect user feedback, iterate

---

## Part 6: Success Metrics (Integrated)

### Quantitative KPIs

| Metric | Current | Target | How Measured |
|--------|---------|--------|--------------|
| **Insight Specificity** | ~20% | 100% | % with $ amounts |
| **Trigger Citation** | 0% | 100% | % citing triggers |
| **UI Contradictions** | ~15/view | 0 | Manual audit |
| **False Positives** | ~40% | <10% | % stable flagged as action |
| **Network Data Growth** | 0 ASINs | 500K+ ASINs | product_snapshots count |
| **Category Coverage** | 0 cats | 300+ cats | Unique categories |
| **Data Quality (HIGH)** | N/A | 80%+ | % categories with HIGH quality |
| **Prediction Accuracy** | Unknown | >80% | Actual vs predicted outcomes |
| **LLM Success Rate** | N/A | >95% | % passing quality gates |
| **User Action Rate** | N/A | >40% | % insights acted on |

### The Network Effect Metric

**Definition:** Prediction accuracy improvement as function of data accumulation

```
Accuracy(t) = Base_Accuracy + (Data_Quality_Score * 0.2)

Where:
- Base_Accuracy = 65% (deterministic baseline)
- Data_Quality_Score = f(ASINs_tracked, Categories_covered, Patterns_observed)
- Max Accuracy = 85% (cap at 85%)

Example:
- Week 1: 100 ASINs, 5 categories, 0 patterns → 65% accuracy
- Week 10: 20K ASINs, 50 categories, 25 patterns → 75% accuracy
- Week 50: 500K ASINs, 200 categories, 150 patterns → 83% accuracy
```

---

## Part 7: Critical Design Decisions & Rationale

### Decision 1: Two-Stage LLM Architecture

**Decision:** Use two separate LLM calls (Classification → Insight)

**Rationale:**
- Better prompt engineering (each call optimized for specific task)
- Easier validation (validate classification, then validate insight)
- Fallback flexibility (deterministic classification + LLM insight)
- Lower latency than single mega-prompt

**Trade-off:** 2x LLM cost, but much higher quality outputs

### Decision 2: Unified Status Taxonomy

**Decision:** Replace conflicting status fields with single `product_status` enum

**Rationale:**
- Eliminates UI contradictions (one source of truth)
- Granular enough for filtering (13 statuses)
- Maps cleanly to priority-based Action Queue
- Backward compatible (still generate `strategic_state` for legacy code)

**Trade-off:** Migration complexity, but long-term simplicity

### Decision 3: Network Intelligence as Separate Subsystem

**Decision:** Build Network Intelligence as independent layer, not embedded in AI Engine

**Rationale:**
- Clean separation of concerns (data vs. analysis)
- Network intelligence can be queried independently
- Easier to test and validate benchmarks
- Scales better (aggregate once, query many times)
- Future-proof (can add more intelligence sources)

**Trade-off:** More complex architecture, but much more flexible

### Decision 4: Trigger Events as First-Class Objects

**Decision:** Store trigger events as structured data, not just text

**Rationale:**
- Enable pattern recognition (find similar triggers)
- Track trigger accuracy (which triggers predict outcomes)
- Allow UI visualization (show timeline of events)
- Support future ML models (train on trigger features)

**Trade-off:** More database tables, but richer analytics

### Decision 5: Quality Validation Gates

**Decision:** Reject LLM outputs that don't meet standards

**Rationale:**
- Prevent vague recommendations from reaching users
- Force specificity (must have $ amounts)
- Maintain user trust (no generic advice)
- Create feedback loop (track what LLM struggles with)

**Trade-off:** ~5-10% of LLM outputs rejected, but 100% meet quality bar

---

## Part 8: Architectural Guarantees

### Guarantee 1: Zero UI Contradictions

**How:** Single `product_status` field used everywhere

**Validation:**
```sql
-- Query: Find products with status mismatch
SELECT * FROM strategic_insights si
JOIN products p ON si.asin = p.asin
WHERE si.product_status != p.status;  -- Should return 0 rows
```

### Guarantee 2: 100% Insight Specificity

**How:** Quality validation gate rejects insights without $ amounts

**Validation:**
```python
# All stored insights must pass this test
assert '$' in insight['recommendation'] or '%' in insight['recommendation']
assert len(insight['trigger_events_cited']) > 0
assert insight['projected_upside_monthly'] > 0 or insight['downside_risk_monthly'] > 0
```

### Guarantee 3: Network Intelligence Growth

**How:** Every search automatically accumulates data

**Validation:**
```sql
-- Query: Data growth rate
SELECT
    DATE_TRUNC('week', snapshot_date) as week,
    COUNT(DISTINCT asin) as new_asins,
    COUNT(DISTINCT category_id) as new_categories
FROM product_snapshots
GROUP BY week
ORDER BY week DESC;
-- Should show continuous growth
```

### Guarantee 4: Causal Reasoning

**How:** LLM prompts REQUIRE citing trigger events and network intelligence

**Validation:**
```python
# All insights must cite sources
assert 'trigger_events_cited' in insight and len(insight['trigger_events_cited']) > 0
assert 'reasoning' in insight and len(insight['reasoning']) > 50
# Reasoning must reference specific data
assert any(keyword in insight['reasoning'].lower() for keyword in [
    'competitor', 'inventory', 'review', 'price', 'median', 'percentile'
])
```

---

## Part 9: Future Enhancements (Post-Launch)

### Enhancement 1: Machine Learning Layer

**Goal:** Train ML models on accumulated data

**Approach:**
- Use trigger events as features
- Use insight outcomes as labels
- Train model to predict: "Will this insight work?"
- Use ML predictions to boost/reduce LLM confidence

**Timeline:** 6 months post-launch (need outcome data)

### Enhancement 2: Cross-Category Learning

**Goal:** Apply patterns from one category to another

**Approach:**
- Detect similar patterns across categories
- Transfer confidence scores
- "Category X has pattern Y with 85% success → suggest testing in Category Z"

**Timeline:** 3 months post-launch (need pattern library)

### Enhancement 3: Real-Time Alerts

**Goal:** Push notifications when triggers detected

**Approach:**
- Background job checks for new triggers every hour
- Push critical alerts (stockout risk, BuyBox loss)
- Email digest of opportunities

**Timeline:** 2 months post-launch

### Enhancement 4: Collaborative Intelligence

**Goal:** Learn from user actions across all users

**Approach:**
- Track: which insights do users act on?
- Track: which actions produce best results?
- Use to improve recommendation ranking

**Timeline:** 4 months post-launch (need action tracking)

---

## Part 10: Checklist for Go-Live

### Pre-Launch Checklist

**Database:**
- [ ] All tables created and indexed
- [ ] RLS policies configured
- [ ] Migration scripts tested
- [ ] Backup strategy in place

**Code:**
- [ ] All three subsystems implemented
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Load tests passing (100+ ASINs in <5s)

**AI/LLM:**
- [ ] LLM prompts finalized and tested
- [ ] Quality validation gates working
- [ ] Fallback logic tested
- [ ] Rate limiting configured

**Data:**
- [ ] Initial seed data loaded (bootstrap network intelligence)
- [ ] Category benchmarks calculated for top 50 categories
- [ ] Test patterns stored

**UI:**
- [ ] Action Queue rendering updated
- [ ] Network intelligence displays working
- [ ] Trigger event visualizations complete
- [ ] User feedback collection hooked up

**Monitoring:**
- [ ] Error tracking (Sentry/etc)
- [ ] Performance monitoring (query latency)
- [ ] LLM call tracking (success rate, cost)
- [ ] Network growth dashboard

**Documentation:**
- [ ] All architecture docs up to date
- [ ] API documentation
- [ ] User guide for new insights
- [ ] Runbook for common issues

---

## Conclusion

This unified architecture transforms ShelfGuard from a monitoring tool into a **self-learning Strategic Intelligence System**.

**Key Innovations:**
1. **Causal Reasoning**: Every recommendation explains WHY
2. **Network Effects**: Gets smarter as more users search
3. **Quality Guarantees**: Zero contradictions, 100% specificity
4. **Feedback Loops**: Track accuracy, improve over time

**The Flywheel:**
```
More Users → More Data → Better Benchmarks → Better AI Predictions →
Better Insights → More User Trust → More Users → [LOOP]
```

**Next Steps:** Begin Phase 1 implementation (database schema + data models).

---

**End of Master Architecture Overview**
