# ShelfGuard Intelligence Pipeline - Quick Start Guide

**Version:** 1.0
**Last Updated:** 2026-01-19

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

1. **Environment Variables** - Add to `.env`:
```bash
KEEPA_API_KEY=your_keepa_key
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

2. **Database Migrations** - Already done! ✅
   - `strategic_insights` table
   - `trigger_events` table
   - `category_intelligence` table
   - Network intelligence tables

### Basic Usage

```python
from src.two_phase_discovery import (
    phase1_seed_discovery,
    phase2_category_market_mapping,
    generate_strategic_intelligence
)

# Step 1: Find products
df_seeds = phase1_seed_discovery(keyword="k-cup coffee", limit=50)

# Step 2: Map market (select a seed product first)
df_market, market_stats = phase2_category_market_mapping(
    category_id=seed_row['category_id'],
    seed_product_title=seed_row['title'],
    seed_asin=seed_row['asin']
)

# Step 3: Generate intelligence for your ASINs
intelligence_results = generate_strategic_intelligence(
    df_market_snapshot=df_market,
    df_weekly=market_stats['df_weekly'],
    portfolio_asins=['B01234ASIN', 'B05678ASIN'],  # YOUR products
    category_context={
        'category_id': seed_row['category_id'],
        'category_name': 'Coffee',
        'category_path': 'Grocery > Beverages > Coffee'
    }
)

# Step 4: Use insights
for insight in intelligence_results:
    print(f"{insight.asin}: {insight.recommendation}")
    print(f"  Upside: ${insight.projected_upside_monthly}/mo")
    print(f"  Status: {insight.product_status.value}")
```

---

## 📊 What You Get

Each `UnifiedIntelligence` object contains:

```python
insight.asin                          # Product ASIN
insight.product_status                # ProductStatus enum (CRITICAL/OPPORTUNITY/WATCH/STABLE)
insight.recommendation                # Specific action ("Raise price to $27.99")
insight.reasoning                     # AI explanation with trigger + network citations
insight.projected_upside_monthly      # $ amount opportunity
insight.downside_risk_monthly         # $ amount risk
insight.net_expected_value            # Net EV
insight.trigger_events                # List of TriggerEvent objects
insight.action_type                   # "repair", "optimize", "harvest", "defend", "expand"
insight.time_horizon_days             # Urgency (1-90 days)
insight.confidence                    # 0-100 confidence score
```

---

## 🎯 Common Use Cases

### Use Case 1: Analyze Your Entire Portfolio

```python
# Get all your tracked ASINs
portfolio_asins = ['B001', 'B002', 'B003', ...]  # Your products

# Run intelligence for all
insights = generate_strategic_intelligence(
    df_market_snapshot=df_market,
    df_weekly=df_weekly,
    portfolio_asins=portfolio_asins,
    category_context=category_context
)

# Filter critical alerts
critical = [i for i in insights if i.product_status.priority == 100]
print(f"🚨 {len(critical)} CRITICAL alerts")
```

### Use Case 2: Action Queue (Sort by Priority)

```python
# Sort by priority (100=CRITICAL → 0=STABLE)
sorted_insights = sorted(
    intelligence_results,
    key=lambda x: x.product_status.priority,
    reverse=True
)

# Display top 10 most urgent
for insight in sorted_insights[:10]:
    print(f"{insight.product_status.display_name}: {insight.asin}")
    print(f"  {insight.recommendation}")
```

### Use Case 3: Opportunity Finder

```python
# Find price optimization opportunities
opportunities = [
    i for i in intelligence_results
    if i.product_status.value == 'opportunity_price_power'
    and i.projected_upside_monthly > 500  # $500+/mo upside
]

for opp in sorted(opportunities, key=lambda x: x.projected_upside_monthly, reverse=True):
    print(f"{opp.asin}: +${opp.projected_upside_monthly:.0f}/mo opportunity")
    print(f"  {opp.recommendation}")
```

### Use Case 4: Query Stored Insights from Database

```python
from src.intelligence_pipeline import get_active_insights_from_db
from supabase import create_client
import os

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Get all active insights with priority >= 50 (WATCH and above)
insights = get_active_insights_from_db(
    supabase=supabase,
    priority_threshold=50
)

print(f"Found {len(insights)} active insights in database")
```

---

## 🌐 Network Intelligence (Automatic)

When you run `generate_strategic_intelligence()` with `enable_network_accumulation=True` (default), the system automatically:

1. **Stores product snapshots** in `product_snapshots` table
2. **Calculates category benchmarks** (median price, reviews, BSR)
3. **Updates brand intelligence** (brand-level aggregates)
4. **Detects market patterns** (e.g., "review advantage → price premium")

This data accumulates over time and makes the AI smarter with every search.

**Check network intelligence:**

```python
from src.network_intelligence import NetworkIntelligence

network = NetworkIntelligence(supabase)

# Get category benchmarks
benchmarks = network.get_category_benchmarks(category_id=16310101)
print(f"Median price: ${benchmarks['median_price']}")
print(f"Median reviews: {benchmarks['median_review_count']}")

# Get competitive position
position = network.get_competitive_position(asin='B01234', category_id=16310101)
print(f"Price vs median: {position['price_vs_median']:+.1f}%")
print(f"Advantages: {position['competitive_advantages']}")
```

---

## 🔍 Trigger Events

Trigger events are the "why" behind insights. Each insight cites specific market changes:

**Available trigger types:**
- `competitor_oos_imminent` - Competitor has <5 units (opportunity)
- `price_war_active` - 3+ price drops in 7 days
- `review_velocity_spike` - 50+ new reviews in 30 days
- `buybox_share_collapse` - BuyBox dropped from >80% to <50%
- `rank_degradation` - BSR worsened by 30%+
- `new_competitor_entered` - Strong new competitor (BSR <10k)

**Access trigger events:**

```python
for insight in intelligence_results:
    print(f"\n{insight.asin}:")
    for trigger in insight.trigger_events[:3]:  # Top 3
        print(f"  🔴 {trigger.event_type} (severity {trigger.severity}/10)")
        print(f"     {trigger.metric_name}: {trigger.baseline_value} → {trigger.current_value}")
```

---

## ⚙️ Configuration Options

### Disable Network Accumulation

```python
intelligence_results = generate_strategic_intelligence(
    ...,
    enable_network_accumulation=False  # Skip data storage
)
```

### Custom LLM Model

Edit `utils/ai_engine_v2.py` and `utils/ai_engine_v3.py`:

```python
response = openai.chat.completions.create(
    model="gpt-4o",  # Change from gpt-4o-mini to gpt-4o for better quality
    ...
)
```

### Adjust Quality Gates

Edit `utils/ai_engine_v3.py` → `validate_insight_quality()`:

```python
# Require confidence >60 instead of >40
if confidence < 60:
    return False
```

---

## 🐛 Troubleshooting

### "Supabase credentials not found"

**Fix:** Add to `.env`:
```bash
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

### "OpenAI API key not found"

**Fix:** Add to `.env`:
```bash
OPENAI_API_KEY=sk-...
```

### "No insights generated"

**Possible causes:**
1. No portfolio ASINs specified → Pass `portfolio_asins=['B01234']`
2. ASINs not in market snapshot → Ensure ASINs exist in `df_market`
3. No historical data → Phase 2 must fetch 90-day history successfully
4. LLM rate limit → Wait and retry

### "Quality gate failed"

This means the AI-generated insight didn't meet quality standards:
- Missing dollar amounts
- No trigger event citations
- No network intelligence citations

The system will use a fallback insight. To see details, check terminal output for quality gate messages.

---

## 📚 Further Reading

- **Complete Architecture:** [docs/MASTER_ARCHITECTURE_OVERVIEW.md](MASTER_ARCHITECTURE_OVERVIEW.md)
- **Integration Guide:** [docs/INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Implementation Progress:** [docs/IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md)
- **AI Engine Details:** [docs/AI_PREDICTIVE_ENGINE_ARCHITECTURE.md](AI_PREDICTIVE_ENGINE_ARCHITECTURE.md)

---

## 🎉 You're Ready!

The intelligence pipeline is now integrated and ready to use. Run the example at [examples/use_intelligence_pipeline.py](../examples/use_intelligence_pipeline.py) to see it in action.

**Next steps:**
1. Update your Command Center UI to display insights
2. Add Action Queue filtered by priority
3. Add Network Intelligence dashboard (optional)

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for UI integration code samples.
