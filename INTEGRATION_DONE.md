# ✅ Intelligence Pipeline - Integration Complete

The Strategic Intelligence System is now fully integrated into your ShelfGuard app.

---

## How to Use

### 1. Run Your ShelfGuard App

```bash
streamlit run apps/shelfguard_app.py
```

### 2. Enable AI Insights

In the Discovery UI, you'll now see a **"🧠 AI Insights"** checkbox in the top right.

Check this box to enable strategic intelligence generation.

### 3. Run Two-Phase Discovery

1. **Phase 1:** Search for products (e.g., "starbucks k-cup")
2. **Phase 2:** Click "Map Market" to discover 100 ASINs
3. **Intelligence Pipeline:** Automatically runs if AI Insights is enabled

### 4. View Insights

After Phase 2 completes, you'll see a new **"🧠 Strategic Intelligence"** section showing:

- **Critical Alerts** 🚨 - Immediate action required
- **Opportunities** 💡 - Revenue opportunities
- **AI Recommendations** - Specific actions with $ amounts
- **Trigger Events** - What caused this insight
- **Confidence Scores** - How reliable the insight is

---

## What Changed

### Files Modified:

1. **[apps/search_to_state_ui.py](apps/search_to_state_ui.py)**
   - Added "🧠 AI Insights" toggle (line 60-68)
   - Intelligence pipeline call after Phase 2 (line 335-358)
   - Insights display section (line 556-599)

### Files Created:

2. **[src/two_phase_discovery.py](src/two_phase_discovery.py)**
   - `generate_strategic_intelligence()` function (line 1136-1254)
   - `_prepare_market_data_for_pipeline()` helper (line 1257-1320)

3. **Core Intelligence Files:**
   - [src/intelligence_pipeline.py](src/intelligence_pipeline.py) - Main orchestrator
   - [src/trigger_detection.py](src/trigger_detection.py) - 6 trigger detectors
   - [src/network_intelligence.py](src/network_intelligence.py) - Network queries
   - [src/data_accumulation.py](src/data_accumulation.py) - Data accumulation
   - [utils/ai_engine_v2.py](utils/ai_engine_v2.py) - Trigger-aware classification
   - [utils/ai_engine_v3.py](utils/ai_engine_v3.py) - Network-aware insights

---

## Environment Variables Required

Add to your `.env` file (if not already present):

```bash
# OpenAI API (required for AI Insights)
OPENAI_API_KEY=sk-...

# Supabase (required for database storage)
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# Keepa API (already configured)
KEEPA_API_KEY=your_keepa_key
```

---

## What Happens Behind the Scenes

When "🧠 AI Insights" is enabled and Phase 2 completes:

1. **Network Accumulation** (Automatic)
   - Stores all 100 discovered products in `product_snapshots` table
   - Calculates category benchmarks (median price, reviews, BSR)
   - Updates brand intelligence
   - Detects market patterns

2. **Trigger Detection** (For Seed Product)
   - Scans 90-day historical data for market changes
   - Detects: inventory issues, price wars, review spikes, BuyBox loss, rank drops, new competitors

3. **AI Classification** (LLM v2)
   - Classifies strategic state based on triggers
   - Provides causal reasoning

4. **AI Insight Generation** (LLM v3)
   - Generates specific recommendation with $ amounts
   - Cites trigger events and network intelligence
   - Calculates upside/downside/net EV

5. **Database Storage**
   - Stores insight in `strategic_insights` table
   - Stores trigger events in `trigger_events` table

---

## Example Output

After enabling AI Insights and running Phase 2, you might see:

```
🧠 Strategic Intelligence

Metrics:
- 🚨 Critical Alerts: 0
- 💡 Opportunities: 1
- Confidence: 85%

💡 B001ABC123 - Price Power Opportunity

Recommendation:
Raise price from $24.99 to $27.99 (+12%). Category median is $28.50 and you have
a review advantage (150 vs 85 median). Test $27.99 for 7 days, monitor conversion rate.

AI Reasoning:
TRIGGER: opportunity_price_power detected (reviews 76% above median). NETWORK: Your
price is 12% below category median ($24.99 vs $28.50) despite having 150 reviews vs
85 median - clear pricing power. Upside: $3 price increase × 150 units/mo = $450/mo.
Risk: May lose 10% of sales = $150/mo downside. Net EV: +$300/mo.

Monthly Upside: $450
Monthly Risk: $150
Net EV: +$300
Action: optimize
Timeframe: 7 days

Trigger Events:
🟡 opportunity_price_power (Severity: 7/10) - review_count: 85.0 → 150.0
```

---

## Optional: Clean Up Example Files

The example files are no longer needed since everything is integrated:

```bash
# These were just for demonstration and can be deleted:
rm examples/use_intelligence_pipeline.py
rm docs/QUICK_START_GUIDE.md
```

Keep the main documentation:
- [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)
- [docs/MASTER_ARCHITECTURE_OVERVIEW.md](docs/MASTER_ARCHITECTURE_OVERVIEW.md)
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

---

## Testing

1. Run the app: `streamlit run apps/shelfguard_app.py`
2. Go to Discovery tab
3. Enable "🧠 AI Insights" checkbox
4. Search for "starbucks k-cup"
5. Select a seed product
6. Click "Map Market"
7. Wait for Phase 2 to complete
8. See strategic intelligence insights appear

---

## Troubleshooting

### "Supabase credentials not found"
- Add `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` to `.env`

### "OpenAI API key not found"
- Add `OPENAI_API_KEY` to `.env`

### No insights appear
- Make sure "🧠 AI Insights" checkbox is enabled
- Check terminal for error messages
- Verify environment variables are set

### Insights are vague
- This happens if quality gates fail
- Check that historical data is available
- System will show fallback insights with lower confidence

---

## What's Next (Optional)

The system is fully functional. Optional enhancements:

1. **Action Queue Dashboard**
   - Show all insights from database sorted by priority
   - See [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) for code samples

2. **Network Intelligence Dashboard**
   - Visualize accumulated category benchmarks
   - Show brand intelligence over time

3. **Multi-ASIN Analysis**
   - Currently analyzes seed product only
   - Could extend to analyze all portfolio ASINs

---

**The intelligence pipeline is ready to use! Just check the "🧠 AI Insights" box and run your search.**
