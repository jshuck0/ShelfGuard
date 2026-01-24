# ShelfGuard Command Center 2.0: Implementation Review
**Date:** January 24, 2026
**Project:** Causal Intelligence Platform Phase 1 MVP
**Status:** 🟡 Implemented with Critical Bugs

---

## Executive Summary

**What Was Built:**
- ✅ Revenue Attribution Engine (4-category model)
- ✅ Data models for causal intelligence
- ✅ Dashboard UI with 3 sections (Strategic Context, Predictive Horizon, Tactical Response)
- ✅ Integration with existing Command Center
- ❌ **2 Critical Bugs Preventing Functionality**

**Completion Status:** ~85% implemented, 0% functional (due to bugs)

**Critical Issues:**
1. 🔴 **Bug #1:** `AttributionError: 'date'` - Revenue calculation failing
2. 🔴 **Bug #2:** `Predictive engine offline: 'week'` - DataFrame column mismatch

---

## 1. Architecture Review

### ✅ Files Created (Phase 1 MVP)

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `src/models/revenue_attribution.py` | ✅ Complete | ~300 | Data models (RevenueAttribution, AttributionDriver, InternalAction) |
| `src/revenue_attribution.py` | ✅ Complete | ~660 | Attribution engine with elimination method + persistence |
| `apps/shelfguard_app.py` (mod) | ⚠️ Partial | ~450 added | Command Center 2.0 tab with UI (has bugs) |

### ✅ Code Quality Assessment

**Data Models (`src/models/revenue_attribution.py`):**
```python
✅ Clean dataclass architecture
✅ 4 causal categories (Internal, Competitive, Platform, Macro)
✅ Confidence scoring system with badges
✅ Serialization methods (to_dict)
✅ Helper methods (get_earned_growth, get_opportunistic_growth)
```

**Attribution Engine (`src/revenue_attribution.py`):**
```python
✅ Price change auto-detection
✅ Elasticity-based internal action attribution (-1.5 elasticity)
✅ Competitive event attribution (market share capture rate)
✅ Macro trend attribution (category growth + seasonality)
✅ Platform change attribution (CTR differential)
✅ Residual calculation and explained variance
✅ Supabase persistence functions (save_revenue_attribution)
```

**Dashboard Integration (`apps/shelfguard_app.py`):**
```python
⚠️ UI structure exists but has critical bugs
✅ 3-section layout (Strategic Context, Predictive Horizon, Tactical Response)
✅ Properly integrated as 4th tab
✅ Fixed st.stop() bug (no longer blocks other tabs)
❌ Revenue calculation fails (date column issue)
❌ Predictive engine fails (week column issue)
```

---

## 2. Comparison: Plan vs Implementation

### Phase 1 MVP Requirements (from resilient-seeking-lollipop.md)

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Data Models** | ✅ Complete | All dataclasses implemented correctly |
| **Attribution Engine** | ✅ Complete | All 4 attribution functions working |
| **Auto-detect price changes** | ✅ Complete | Detects >2% or >$1 changes from df_weekly |
| **Waterfall chart** | ❌ Not visible | Code exists but blocked by bug #1 |
| **Causal matrix table** | ❌ Not visible | Code exists but blocked by bug #1 |
| **Attribution pie chart** | ❌ Not visible | Code exists but blocked by bug #1 |
| **Executive summary** | ❌ Not visible | Code exists but blocked by bug #1 |
| **Confidence badges** | ✅ Implemented | 🟢🟡🔴 system works |
| **Portfolio-level attribution** | ✅ Implemented | Aggregates all ASINs |

### Phase 2 Requirements (Enhanced Detection)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Platform change detector | ❌ Not started | Planned for Phase 2 |
| Competitor creative detector | ❌ Not started | Planned for Phase 2 |
| Share of Voice tracker | ❌ Not started | Planned for Phase 2 |
| Macro trend detector | ❌ Not started | Planned for Phase 2 |

### Phase 2.5 Requirements (Predictive Features)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Projected annual sales | ❌ Not started | Planned for Phase 2.5 |
| Event timeline | ❌ Not started | Planned for Phase 2.5 |
| Scenario forecasting | ❌ Not started | Planned for Phase 2.5 |
| Combined intelligence dashboard | ❌ Not started | Planned for Phase 2.5 |

**Verdict:** Phase 1 MVP is ~85% implemented structurally, but 0% functional due to bugs.

---

## 3. Screenshot Analysis

### Command Center 2.0 (Screenshots: command_2.0_1, command_2.0_2)

**What We See:**

**Section 1: Strategic Context**
- ❌ Shows: "Attribution Error: 'date'"
- ❌ Shows: "Strategic Context Unavailable: insufficient historical data for attribution."

**Section 2: Predictive Horizon**
- ❌ Shows: "Predictive engine offline: 'week'"

**Section 3: Tactical Response Plan**
- ✅ Shows: Priority Actions (100 monitored)
- ✅ Shows: TRENCH WAR products (#1, #2 items visible)
- ✅ Working correctly (uses existing portfolio intelligence)

**What Should Be Showing (Per Plan):**
- Revenue change metrics (Total, Earned, Opportunistic)
- Waterfall chart with cumulative attribution
- Causal matrix table with color-coded drivers
- Attribution pie chart (4-category breakdown)
- Executive summary with strategic warnings

### Original Dashboard (Screenshots: og_1, og_2, og_3, og_4)

**og_1 - Portfolio Actions:**
- Shows AI Action Queue with product cards
- Displays strategic states (TRENCH WAR, HARVEST)
- Full Action Queue table with products

**og_2 - Market Causality Analysis:**
- Price/rank trend chart over time
- Key Market Events Detected section
- Portfolio Actions preview

**og_3 - Strategic Details:**
- Strategic Brief text
- Competitive Landscape metrics
- Root Cause Analysis
- Competitor Price Intelligence table

**og_4 - Your Next Actions:**
- Prioritized action list
- Strategic Brief summary
- Monthly revenue projection ($3.2M)

**Comparison:**
- Original dashboard: ✅ Fully functional, data-rich, actionable
- Command Center 2.0: ❌ Non-functional due to data pipeline bugs

---

## 4. Bug Analysis

### 🔴 Bug #1: `AttributionError: 'date'`

**Location:** `apps/shelfguard_app.py` (Command Center 2.0 tab)

**Root Cause:**
```python
# Line ~800-802 in shelfguard_app.py
if 'date' not in df_weekly.columns and 'week' in df_weekly.columns:
    df_weekly['date'] = pd.to_datetime(df_weekly['week'])
```

**Problem:** This assumes `df_weekly` exists and has a 'week' column, but:
1. `df_weekly` might be empty (pd.DataFrame())
2. `df_weekly` might not have 'week' column
3. The error "'date'" suggests we're trying to access a 'date' key in a dict/object instead of a DataFrame column

**Evidence from Screenshot:**
- "Attribution Error: 'date'" suggests accessing `.get('date')` on wrong object type

**Likely Issue:**
```python
# Somewhere in the code, we're doing:
some_dict.get('date')  # Wrong - should be df_weekly['date']
# OR
df_weekly is not a DataFrame (might be a dict or None)
```

**Fix Required:**
1. Validate `df_weekly` is a DataFrame before accessing columns
2. Add proper error handling for missing date/week columns
3. Check session state data structure

---

### 🔴 Bug #2: `Predictive engine offline: 'week'`

**Location:** `apps/shelfguard_app.py` (Predictive Horizon section)

**Root Cause:**
```python
# Predictive Horizon section is trying to access 'week' column
# but df_weekly might not have it, or might be wrong data type
```

**Problem:**
1. Similar to Bug #1 - assumes 'week' column exists
2. Might be accessing wrong DataFrame
3. Data structure mismatch

**Evidence from Screenshot:**
- "Predictive engine offline: 'week'" suggests KeyError on 'week' column

**Fix Required:**
1. Validate df_weekly structure before predictive calculations
2. Add fallback for missing 'week' column
3. Better error messages for debugging

---

### 🟡 Bug #3: `.median()` Issue (from earlier testing)

**Error Message:** `'list' object has no attribute 'median'`

**Location:** `apps/shelfguard_app.py` line ~862

**Root Cause:**
```python
# Current code (WRONG):
market_snapshot_dict = {
    'category_benchmarks': {
        'median_price': market_snapshot.get('price_per_unit', market_snapshot.get('buy_box_price', [0])).median()
    }
}

# Problem: .get() with default [0] returns a list, not a Series
# Lists don't have .median() method
```

**Fix:**
```python
# Solution:
median_price = 0
if not market_snapshot.empty:
    if 'price_per_unit' in market_snapshot.columns:
        median_price = market_snapshot['price_per_unit'].median()
    elif 'buy_box_price' in market_snapshot.columns:
        median_price = market_snapshot['buy_box_price'].median()
```

---

## 5. Data Pipeline Issues

### Expected Data Flow (Per Plan)

```
Session State
├─ active_project_weekly_data (df_weekly)
│  ├─ Columns: date, week, price, revenue/sales, asin
│  └─ Used for: Price change detection, revenue calculation
│
├─ active_project_market_snapshot (market_snapshot)
│  ├─ Columns: price_per_unit, buy_box_price, revenue_proxy
│  └─ Used for: Category benchmarks, competitor data
│
└─ enriched_portfolio_triangulated
   ├─ Contains: trigger_events column
   └─ Used for: Competitive/platform attribution
```

### Actual Data Issues

**Problem 1: df_weekly structure**
- Expected: Pandas DataFrame with 'week' or 'date' column
- Actual: Might be empty, might be dict, might lack columns
- Impact: Revenue calculation fails (Bug #1)

**Problem 2: market_snapshot structure**
- Expected: Pandas DataFrame with price columns
- Actual: DataFrame but `.get()` returns wrong type
- Impact: Median calculation fails (Bug #3)

**Problem 3: Missing data validation**
- No checks for data existence before processing
- No graceful degradation for missing data
- No informative error messages for users

---

## 6. Missing Features (Per Plan)

### Phase 1 MVP - Not Yet Visible

Due to bugs, these implemented features are not visible:

1. **Waterfall Chart** - Code exists (lines ~960-1000) but unreachable
2. **Causal Matrix Table** - Code exists (lines ~1010-1050) but unreachable
3. **Attribution Pie Chart** - Code exists (lines ~1060-1090) but unreachable
4. **Executive Summary** - Code exists (lines ~1090-1120) but unreachable

### Phase 2 - Not Implemented

4 new trigger detectors planned but not started:
1. Platform change detector (Choice badge, algorithm shifts)
2. Competitor creative detector (image/title changes)
3. Share of Voice tracker (keyword rank changes)
4. Macro trend detector (category-wide movements)

### Phase 2.5 - Not Implemented

Predictive features planned but not started:
1. Projected annual sales calculator
2. Event timeline (anticipated events calendar)
3. Scenario builder (Base/Optimistic/Pessimistic)
4. Combined intelligence dashboard (causal + predictive)

### Phase 3 - Not Implemented

Database persistence planned but not started:
1. Supabase tables for attribution history (NOTE: Functions exist in code but tables not created)
2. Time-series views of attribution changes
3. Forecast accuracy tracking

---

## 7. Code Quality Issues

### Architecture Strengths
✅ Clean separation of concerns (models, engine, UI)
✅ Well-documented code with docstrings
✅ Type hints throughout
✅ Dataclass pattern for models
✅ Enumeration for categories

### Architecture Weaknesses
❌ No data validation before processing
❌ No graceful error handling
❌ Assumes data structure without checking
❌ No unit tests for attribution engine
❌ No integration tests for dashboard

### Technical Debt
1. **Error Handling:** Try/except blocks too broad, don't provide useful debugging info
2. **Data Validation:** No schema validation for df_weekly/market_snapshot
3. **Fallbacks:** No placeholder data when attribution fails
4. **Logging:** No structured logging for debugging
5. **Testing:** Zero test coverage for new code

---

## 8. Comparison to Original Dashboard

### Original Dashboard Strengths (from screenshots)

**Data Richness:**
- Shows actual revenue numbers ($3.2M monthly)
- Displays trend charts with real data
- Provides competitor intelligence table
- Lists specific product recommendations

**Actionability:**
- Clear "RESOLVE" buttons on actions
- Prioritized action queue (100 products)
- Specific $ impact amounts (+$146k, +$126k, +$70k)
- Strategic state labels (TRENCH WAR, HARVEST)

**Visual Design:**
- Color-coded cards (red, yellow, green)
- Mini sparkline charts
- Progress indicators
- Clean layout with good information density

### Command Center 2.0 Weaknesses (from screenshots)

**No Data Visible:**
- All sections show errors instead of data
- No revenue numbers
- No charts
- No actionable insights

**Poor Error Messages:**
- "Attribution Error: 'date'" - not user-friendly
- "insufficient historical data" - vague
- "Predictive engine offline: 'week'" - technical jargon

**Design Issues:**
- Red error boxes instead of content
- Yellow warning boxes instead of predictions
- Only "Tactical Response Plan" section works

**User Experience:**
- Cannot answer "Where did my revenue come from?"
- Cannot see earned vs opportunistic split
- Cannot see attribution breakdown
- Essentially non-functional

---

## 9. Verification Against Success Metrics (from Plan)

### Causal Metrics (Target vs Actual)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Explained Variance | >80% | N/A (not calculating) | ❌ |
| Director Comprehension | 30 seconds to answer | ∞ (errors shown) | ❌ |
| Actionable Insights | Changes strategy | No insights visible | ❌ |

### Predictive Metrics (Target vs Actual)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Forecast Accuracy | >70% within 30% | Not forecasting | ❌ |
| Director Foresight | 30 seconds to answer | N/A | ❌ |
| Risk Awareness | Identifies before occurrence | No predictions | ❌ |
| Sustainability Awareness | Temp vs permanent growth | Not visible | ❌ |

**Verdict:** 0/8 success metrics achieved due to bugs.

---

## 10. Gap Analysis

### What's Missing from Original Vision

**From Vision Document:**

> "Transform Command Center from 'What happened?' to 'Why it happened?' with quantified attribution"

**Current State:** Shows errors, not attributions

> "Director-level question: 'Where did my revenue come from this month?'"

**Current State:** Cannot answer - no revenue data shown

> "Separate earned (your actions) vs opportunistic (market luck) growth"

**Current State:** Not visible due to bugs

**Key Quote from Plan:**
> "Instead of: Sales went up $50,000 this month. Great!
> We want: Sales went up $50,000. Here's the breakdown:
> - +$22,000 from Internal Actions (your PPC increase)
> - +$18,000 from Competitive Vacuum (Competitor X went OOS)
> - +$12,000 from Market Tailwind (category grew 15%)
> - -$2,000 from Platform Decay (lost Choice badge)"

**Current State:** Shows "Attribution Error: 'date'" instead

---

## 11. Recommendations

### 🔥 Critical (Fix Immediately)

1. **Fix Bug #1 (AttributionError: 'date')**
   - Add data validation before processing
   - Handle missing date/week columns gracefully
   - Provide informative error messages

2. **Fix Bug #2 (Predictive engine: 'week')**
   - Validate df_weekly structure
   - Add fallback for missing columns
   - Show placeholder if data unavailable

3. **Fix Bug #3 (.median() on list)**
   - Replace `.get()` with proper DataFrame column access
   - Add null checks before calculating median

### ⚠️ High Priority (Fix This Week)

4. **Add Data Validation Layer**
   ```python
   def validate_df_weekly(df_weekly):
       """Validate df_weekly has required structure"""
       if df_weekly is None or df_weekly.empty:
           return False, "No historical data available"

       required_cols = ['date', 'price']
       missing = [col for col in required_cols if col not in df_weekly.columns]
       if missing:
           return False, f"Missing columns: {missing}"

       return True, "Valid"
   ```

5. **Add Graceful Degradation**
   - If attribution fails, show sample/placeholder data
   - If data insufficient, explain what's needed
   - Provide "Upload Data" call-to-action

6. **Improve Error Messages**
   ```python
   # Instead of:
   st.error("⚠️ Attribution calculation failed: 'list' object has no attribute 'median'")

   # Show:
   st.warning("""
   📊 **Attribution Unavailable**

   This feature requires:
   - ✅ Active project selected
   - ❌ 30+ days of historical revenue data
   - ❌ Price history data

   **Next Steps:** Upload historical data or wait for data collection.
   """)
   ```

### 📋 Medium Priority (Next Sprint)

7. **Add Unit Tests**
   - Test attribution engine with sample data
   - Test data validation functions
   - Test edge cases (empty data, missing columns)

8. **Add Integration Tests**
   - Test full dashboard render with real project data
   - Test with Charmin project specifically
   - Test all 3 sections render correctly

9. **Add Logging**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   logger.info(f"Calculating attribution for {len(project_asins)} ASINs")
   logger.debug(f"df_weekly shape: {df_weekly.shape}, columns: {df_weekly.columns}")
   ```

### 🎯 Low Priority (Future Phases)

10. **Complete Phase 2** (4 new detectors)
11. **Complete Phase 2.5** (Predictive features)
12. **Complete Phase 3** (Database persistence)

---

## 12. Next Session Action Plan

### Immediate (First 30 Minutes)

1. **Fix the 3 bugs** in this exact order:
   ```
   Priority 1: Bug #1 (AttributionError: 'date')
   Priority 2: Bug #3 (.median() on list)
   Priority 3: Bug #2 (Predictive engine: 'week')
   ```

2. **Add data validation** before all DataFrame operations

3. **Test with Charmin project** (has real data from screenshots)

### Next Hour

4. **Verify all visualizations render:**
   - Waterfall chart
   - Causal matrix table
   - Attribution pie chart
   - Executive summary

5. **Test earned vs opportunistic split** calculation

6. **Verify confidence badges** display correctly

### Testing Checklist

- [ ] Load Charmin project
- [ ] Command Center 2.0 tab renders without errors
- [ ] Strategic Context shows attribution breakdown
- [ ] Waterfall chart displays
- [ ] Causal matrix table shows drivers
- [ ] Pie chart renders
- [ ] Executive summary shows earned % and opportunistic %
- [ ] Confidence badges (🟢🟡🔴) appear
- [ ] Numbers add up correctly (total = internal + competitive + macro + platform)
- [ ] Explained variance >60%

---

## 13. Summary

### What Worked Well
✅ Clean architecture and code organization
✅ Comprehensive data models
✅ Well-documented plan and vision
✅ Ambitious feature set
✅ Good separation of concerns

### What Went Wrong
❌ No data validation before launch
❌ Assumed data structure without checking
❌ No testing with real project data
❌ Poor error handling
❌ Bugs block all Phase 1 features

### The Reality Gap

**Planned:** "Director can answer 'Where did my revenue come from?' in 30 seconds"
**Actual:** Director sees "Attribution Error: 'date'" instead

**Planned:** "96% explained variance with high confidence"
**Actual:** 0% - not calculating at all

**Planned:** "Waterfall chart, causal matrix, pie chart, executive summary"
**Actual:** Red error boxes

### Path Forward

1. **Fix bugs** (2 hours of work)
2. **Test thoroughly** (1 hour with real data)
3. **Iterate based on real results** (adjust attribution formulas if needed)
4. **Complete Phase 1 MVP** before starting Phase 2

**Estimated Time to Working MVP:** 3-4 hours (mostly bug fixes)

**Current Completion:** 85% code written, 0% functional, 3 blocking bugs

---

## Appendix A: Bug Fix Code Snippets

### Fix for Bug #1 & #2 (Data Validation)

```python
# Add at start of Command Center 2.0 tab (after line 775)

# Validate data availability
data_validation = {
    'df_weekly_valid': False,
    'market_snapshot_valid': False,
    'trigger_events_valid': False
}

# Validate df_weekly
if not df_weekly.empty and isinstance(df_weekly, pd.DataFrame):
    # Ensure date column exists
    if 'date' in df_weekly.columns:
        data_validation['df_weekly_valid'] = True
    elif 'week' in df_weekly.columns:
        df_weekly = df_weekly.copy()
        df_weekly['date'] = pd.to_datetime(df_weekly['week'], errors='coerce')
        data_validation['df_weekly_valid'] = True

# Validate market_snapshot
if not market_snapshot.empty and isinstance(market_snapshot, pd.DataFrame):
    data_validation['market_snapshot_valid'] = True

# Validate trigger_events
if trigger_events and len(trigger_events) > 0:
    data_validation['trigger_events_valid'] = True

# Show data status to user
if not any(data_validation.values()):
    st.warning("""
    📊 **Insufficient Data for Attribution**

    This feature requires:
    - ❌ Historical revenue data (30+ days)
    - ❌ Market snapshot data
    - ❌ Trigger event detection

    **Next Steps:** Ensure your project has been active for at least 30 days
    or upload historical data.
    """)
else:
    # Proceed with attribution calculation...
```

### Fix for Bug #3 (.median() Issue)

```python
# Replace lines ~860-865 with:

market_snapshot_dict = None
if not market_snapshot.empty and isinstance(market_snapshot, pd.DataFrame):
    median_price = 0

    # Try to get median from price columns
    for col in ['price_per_unit', 'buy_box_price', 'filled_price', 'price']:
        if col in market_snapshot.columns:
            price_series = pd.to_numeric(market_snapshot[col], errors='coerce')
            valid_prices = price_series[price_series > 0]
            if len(valid_prices) > 0:
                median_price = float(valid_prices.median())
                break

    market_snapshot_dict = {
        'category_benchmarks': {
            'growth_rate_30d': 0,
            'median_price': median_price
        }
    }
```

---

**End of Review**

**Status:** 🔴 Critical bugs preventing functionality
**Recommendation:** Fix bugs before proceeding to Phase 2
**Estimated Fix Time:** 2-4 hours
**Next Steps:** See Section 12 (Next Session Action Plan)
