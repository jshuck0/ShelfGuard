# ShelfGuard Search-to-State Architecture

## Overview

The **Search-to-State** model transforms ShelfGuard from a manual ASIN-upload tool to a dynamic market discovery platform. Users can search for brands or categories, instantly prune results to the top revenue-driving ASINs, and "pin" them as persistent projects with proactive monitoring.

---

## Architecture Components

### Phase 1: Discovery Engine (`src/discovery.py`)

**Purpose**: Search Keepa → Calculate Revenue → Prune to 90%

#### Key Functions:
- `search_keepa_market(query, search_type, limit=500)` - Fetches top 500 ASINs from Keepa Product Finder API
- `calculate_revenue_proxy(products)` - Computes `Monthly Units × Price` for each ASIN
- `prune_to_90_percent(df)` - Applies 90% Revenue Rule (find minimum ASIN set that captures 90% of total revenue)
- `execute_market_discovery(query, search_type)` - Full pipeline orchestrator

#### Performance Optimizations:
- **24-hour caching** via `@st.cache_data(ttl=86400)` - prevents redundant API calls when multiple users search the same brand
- Batched Keepa requests (100 ASINs per call)

#### Example Usage:
```python
from src.discovery import execute_market_discovery

# Search for "Starbucks" brand
market_snapshot, stats = execute_market_discovery(
    query="Starbucks",
    search_type="brand",
    limit=500
)

print(f"Pruned {stats['total_asins']} → {stats['pruned_asins']} ASINs")
print(f"Revenue Captured: {stats['revenue_captured_pct']:.1f}%")
```

---

### Phase 2: Persistence Layer (`src/persistence.py`)

**Purpose**: Pin discovered ASINs → Create persistent projects → Apply Mission Profiles

#### Key Functions:
- `pin_to_state(asins, project_name, mission_type, user_id, metadata)` - Creates new project in Supabase
- `get_mission_profile_config(mission_type)` - Returns alert priority weights for:
  - **"bodyguard"** - Defensive focus (price protection, Buy Box)
  - **"scout"** - Offensive focus (new entrants, rising stars)
  - **"surgeon"** - Efficiency focus (review gaps, ad waste)
- `load_user_projects(user_id)` - Fetches all projects for a user (RLS-protected)
- `load_project_asins(project_id)` - Gets tracked ASINs for a project

#### Row Level Security (RLS):
All tables (`projects`, `tracked_asins`, `historical_metrics`) are protected by RLS policies:
```sql
-- Users can only see their own projects
CREATE POLICY "Users can view their own projects"
    ON projects FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);
```

Anonymous users (`user_id = NULL`) can create projects for demo/testing.

#### Example Usage:
```python
from src.persistence import pin_to_state

# Pin pruned ASINs to a new project
project_id = pin_to_state(
    asins=market_snapshot["asin"].tolist(),
    project_name="Starbucks K-Cups Monitoring",
    mission_type="bodyguard",  # Defensive focus
    user_id=None,  # Anonymous for now
    metadata=stats  # Store discovery stats
)

print(f"✅ Project created: {project_id}")
```

---

### Phase 3: Historical Backfill (`src/backfill.py`)

**Purpose**: Fetch 90 days of Price & BSR history → Populate `historical_metrics` table → Enable Day 1 charts

#### Key Functions:
- `fetch_90day_history(asins, domain=1)` - Fetches full Keepa history for up to 100 ASINs
- `keepa_minutes_to_unix(keepa_minutes)` - Converts Keepa time format:
  ```
  Unix Time (ms) = (KeepaMinutes + 21,564,000) × 60,000
  ```
- `parse_historical_timeseries(product, csv_index, metric_name)` - Extracts BSR, Buy Box, Amazon Price, FBA Price
- `build_historical_metrics(products)` - Merges all time series into unified DataFrame
- `upsert_historical_metrics(df, project_id, supabase)` - Batch-inserts records (500 per chunk)
- `execute_backfill(project_id, asins, run_async=True)` - **Runs in background thread** to keep UI responsive

#### Asynchronous Execution:
By default, backfill runs in a background thread:
```python
from src.backfill import execute_backfill

execute_backfill(
    project_id=project_id,
    asins=pruned_asins,
    run_async=True  # Non-blocking
)

st.info("🔄 Backfill started in background...")
```

#### Performance:
- ~5-10 seconds for 100 ASINs (depends on Keepa API latency)
- Chunked inserts avoid Supabase timeouts
- Only fetches 90 days (not full 36-month history) to save API tokens

---

### Phase 4: Outcome Triangulation (`src/recommendations.py`)

**Purpose**: Generate proactive alerts based on historical velocity + mission profile priorities

#### Alert Types:

| Alert Type | Detection Logic | Severity |
|------------|----------------|----------|
| **Volume Stealer** | BSR improved >20% + Price dropped >10% in 7 days | High |
| **Efficiency Gap** | Top 20 ASIN with reviews <50% of category average | Medium |
| **New Entrant** | New ASIN entered market with BSR <50k | Medium |
| **Buy Box Loss** | BB share dropped <50% in 7 days | High (TODO) |

#### Key Functions:
- `calculate_bsr_velocity(df_metrics, lookback_days=7)` - BSR % change over 7 days
- `calculate_price_delta(df_metrics, lookback_days=7)` - Price % change over 7 days
- `detect_volume_stealers(df_metrics)` - Identifies aggressive competitors
- `detect_efficiency_gaps(df_current)` - Finds review/content opportunities
- `generate_resolution_cards(df_metrics, df_current, mission_type)` - Master orchestrator
- `render_resolution_card(alert)` - Streamlit UI component

#### Mission Profile Prioritization:
Alerts are scored based on mission profile:
```python
# Bodyguard: Prioritize price protection
priorities = {
    "price_undercut": 1.0,  # Highest
    "buybox_loss": 1.0,
    "volume_stealer": 0.8,
    "review_gaps": 0.3      # Lowest
}
```

#### Example Usage:
```python
from src.recommendations import generate_resolution_cards, render_resolution_card

# Generate alerts
alerts = generate_resolution_cards(
    df_metrics=historical_data,  # From historical_metrics table
    df_current=current_week_snapshot,
    mission_type="bodyguard"
)

# Display in UI
for alert in alerts[:5]:  # Top 5 by priority
    render_resolution_card(alert)
```

---

## Database Schema

Run `schemas/search_to_state.sql` in Supabase SQL Editor to create:

### Tables:
1. **`projects`** - User-created monitoring projects
   - Columns: `id`, `user_id`, `project_name`, `mission_type`, `asin_count`, `metadata`
   - RLS: Users can only see their own projects

2. **`tracked_asins`** - Many-to-many link between projects and ASINs
   - Columns: `project_id`, `asin`, `added_at`, `is_active`
   - Unique constraint: `(project_id, asin)`

3. **`historical_metrics`** - 90-day Price & BSR history
   - Columns: `project_id`, `asin`, `datetime`, `sales_rank`, `buy_box_price`, `amazon_price`, `new_fba_price`
   - Unique constraint: `(project_id, asin, datetime)`
   - Indexed on `(project_id, asin, datetime DESC)` for fast time-series queries

4. **`resolution_cards`** (Optional) - Persisted alerts
   - Columns: `project_id`, `alert_type`, `severity`, `asin`, `title`, `message`, `action`, `is_dismissed`

---

## UI Integration

### Discovery UI (New):
```python
import streamlit as st
from src.discovery import execute_market_discovery
import plotly.express as px

# Search input
query = st.text_input("🔍 Search Brand or Category", placeholder="e.g., Starbucks")

if query:
    with st.spinner("Scanning market..."):
        market_snapshot, stats = execute_market_discovery(query, "brand")

    # Display stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total ASINs", stats["total_asins"])
    col2.metric("Pruned ASINs", stats["pruned_asins"])
    col3.metric("Revenue Captured", f"{stats['revenue_captured_pct']:.1f}%")

    # Visualizations
    tab1, tab2 = st.tabs(["Market Share", "Price vs BSR"])

    with tab1:
        # Donut chart
        fig = px.pie(
            market_snapshot,
            values="revenue_proxy",
            names="title",
            title="Revenue Distribution (90% Set)"
        )
        st.plotly_chart(fig)

    with tab2:
        # Scatter plot
        fig = px.scatter(
            market_snapshot,
            x="bsr",
            y="price",
            size="monthly_units",
            hover_data=["title"],
            title="Price vs BSR",
            labels={"bsr": "Sales Rank", "price": "Price ($)"}
        )
        st.plotly_chart(fig)
```

### Pin to State UI:
```python
from src.persistence import pin_to_state, get_mission_profile_config
from src.backfill import execute_backfill

# Mission profile selector
mission_type = st.radio(
    "Choose Mission Profile:",
    ["bodyguard", "scout", "surgeon"],
    format_func=lambda x: get_mission_profile_config(x)["name"]
)

project_name = st.text_input("Project Name", value=f"{query} Monitoring")

if st.button("📌 Pin to State"):
    # Create project
    project_id = pin_to_state(
        asins=market_snapshot["asin"].tolist(),
        project_name=project_name,
        mission_type=mission_type,
        user_id=None,  # TODO: Get from auth
        metadata=stats
    )

    # Trigger backfill
    execute_backfill(project_id, market_snapshot["asin"].tolist(), run_async=True)

    st.success(f"✅ Project created! Historical backfill in progress...")
```

### Resolution Cards UI:
```python
from src.recommendations import generate_resolution_cards, render_resolution_card

# Fetch historical metrics for project
from src.persistence import create_supabase_client
supabase = create_supabase_client()

result = supabase.table("historical_metrics").select("*").eq(
    "project_id", project_id
).execute()

df_metrics = pd.DataFrame(result.data)

# Generate alerts
alerts = generate_resolution_cards(
    df_metrics=df_metrics,
    df_current=current_week_data,  # From existing engine
    mission_type=mission_type
)

# Render
st.subheader("🎯 Resolution Cards")
for alert in alerts:
    render_resolution_card(alert)
```

---

## Deployment Checklist

### 1. Database Setup
- [ ] Run `schemas/search_to_state.sql` in Supabase SQL Editor
- [ ] Verify RLS policies are active
- [ ] Test anonymous access (`user_id = NULL`)
- [ ] Add indexes if query performance degrades

### 2. Environment Variables
Ensure `.env` contains:
```env
KEEPA_API_KEY=your_keepa_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
```

### 3. Dependencies
Add to `requirements.txt`:
```
streamlit
supabase
pandas
numpy
plotly
requests
python-dotenv
```

### 4. Performance Optimizations
- **API Conservation**: `@st.cache_data(ttl=86400)` on search queries (24h cache)
- **Thread Safety**: Backfill runs in background thread (non-blocking)
- **Visual Consistency**: Use existing CSS theme variables for Plotly charts

### 5. Testing
```bash
# Test discovery
python -c "from src.discovery import execute_market_discovery; execute_market_discovery('Starbucks', 'brand')"

# Test persistence
python -c "from src.persistence import pin_to_state; pin_to_state(['B07GMLSQG5'], 'Test Project', 'bodyguard')"

# Test backfill
python -c "from src.backfill import execute_backfill; execute_backfill('test-uuid', ['B07GMLSQG5'], run_async=False)"
```

---

## Migration Path

### From Current State → Search-to-State

1. **Keep Existing**: `apps/data.py`, `apps/engine.py`, `apps/finance.py` remain unchanged
2. **New Entry Point**: Add search UI to `apps/shelfguard_app.py` (sidebar or new tab)
3. **Gradual Rollout**: Run both systems in parallel:
   - Existing users: Continue using `get_all_data()` (pulls from `keepa_weekly_rows`)
   - New users: Use discovery → pin to state → backfill
4. **Future**: Migrate `keepa_weekly_rows` data into project-scoped `historical_metrics`

---

## API Token Economics

### Keepa Token Usage:
| Operation | Tokens | Frequency |
|-----------|--------|-----------|
| Discovery (500 ASINs) | ~500 | Per search (cached 24h) |
| 90-Day Backfill (100 ASINs) | ~1,000 | Per project creation |
| Weekly Refresh (100 ASINs) | ~100 | Weekly |

**Optimization**: With 100 premium seats, if each user creates 1 project/week:
- Weekly cost: 100 projects × 1,100 tokens = 110,000 tokens
- With 24h search caching, actual discovery cost ≈ 10,000 tokens (80% hit rate)
- **Total**: ~120,000 tokens/week

---

## Future Enhancements

1. **Real-Time Monitoring**: Keepa webhooks for price/BSR changes
2. **Competitor Tracking**: Auto-add new entrants to tracked_asins
3. **Forecast Integration**: Use `apps/engine.py` demand forecast in resolution cards
4. **Email Alerts**: Send high-severity alerts via Supabase Edge Functions
5. **Export to CSV**: Download resolution cards as action items

---

## Support

For issues or questions:
- Check existing `apps/` modules for integration patterns
- Review Keepa API docs: https://keepa.com/#!discuss/t/product-object/116
- Supabase RLS guide: https://supabase.com/docs/guides/auth/row-level-security

---

**Built with Claude Code** 🛡️
