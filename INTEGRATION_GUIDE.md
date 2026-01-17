# Search-to-State Integration Guide

## Quick Start (15 Minutes)

Follow these steps to integrate Search-to-State into the existing ShelfGuard dashboard.

---

## Step 1: Database Setup (5 minutes)

1. Open Supabase SQL Editor
2. Copy the entire contents of `schemas/search_to_state.sql`
3. Run the script
4. Verify tables were created:
   ```sql
   SELECT table_name FROM information_schema.tables
   WHERE table_schema = 'public'
   AND table_name IN ('projects', 'tracked_asins', 'historical_metrics', 'resolution_cards');
   ```

Expected output: 4 tables

---

## Step 2: Install Dependencies (2 minutes)

```bash
pip install -r requirements_clean.txt
```

Or manually:
```bash
pip install streamlit supabase pandas numpy plotly requests python-dotenv openai
```

---

## Step 3: Update shelfguard_app.py (5 minutes)

Add the new UI components to the existing dashboard:

### Option A: Add as New Tabs (Recommended)

Open `apps/shelfguard_app.py` and find the tab navigation section (around line 280). Update it:

```python
# BEFORE (2 tabs):
tab1, tab2 = st.tabs(["🎯 AI Action Queue", "🖼️ Visual Audit"])

# AFTER (4 tabs):
from apps.search_to_state_ui import (
    render_discovery_ui,
    render_project_dashboard,
    render_project_selector
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 AI Action Queue",
    "🖼️ Visual Audit",
    "🔍 Market Discovery",  # NEW
    "📂 My Projects"         # NEW
])

with tab3:
    render_discovery_ui()

with tab4:
    project_id = render_project_selector()
    if project_id:
        render_project_dashboard(project_id)
```

### Option B: Add to Sidebar (Alternative)

Add to the sidebar (after date range selector):

```python
# In sidebar section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Quick Discovery")

with st.sidebar.expander("Search Market"):
    query = st.text_input("Brand/Category", placeholder="e.g., Starbucks")

    if query and st.button("Search"):
        # Redirect to discovery tab
        st.session_state["discovery_query"] = query
        st.session_state["active_tab"] = "discovery"
```

---

## Step 4: Test the Integration (3 minutes)

1. Start the Streamlit app:
   ```bash
   streamlit run apps/shelfguard_app.py
   ```

2. Navigate to the "Market Discovery" tab

3. Search for a brand (e.g., "Starbucks")

4. Verify:
   - [ ] Market snapshot loads (charts + table)
   - [ ] Stats cards display correctly
   - [ ] "Create Project" button works
   - [ ] Backfill starts (check console for progress)

5. Switch to "My Projects" tab:
   - [ ] Created project appears in dropdown
   - [ ] Historical charts populate after backfill completes (~30 seconds)

---

## Step 5: Configure Environment (1 minute)

Ensure your `.streamlit/secrets.toml` file contains:

```toml
[default]
url = "https://your-project.supabase.co"
key = "your-anon-key"

[openai]
OPENAI_API_KEY = "sk-..."
```

And `.env` file contains:

```env
KEEPA_API_KEY=your_keepa_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key
```

---

## Architecture Overview

### New Files Created:
```
ShelfGuard/
├── src/
│   ├── __init__.py
│   ├── discovery.py          # Phase 1: Keepa search + 90% pruning
│   ├── persistence.py         # Phase 2: Project creation + RLS
│   ├── backfill.py            # Phase 3: 90-day historical fetch
│   └── recommendations.py     # Phase 4: Alert generation
├── apps/
│   └── search_to_state_ui.py  # Streamlit UI components
├── schemas/
│   └── search_to_state.sql    # Database schema
├── SEARCH_TO_STATE_README.md  # Full documentation
└── INTEGRATION_GUIDE.md       # This file
```

### Existing Files (Unchanged):
- `apps/data.py` - Still used for legacy `keepa_weekly_rows` data
- `apps/engine.py` - Still runs 36-month velocity analysis
- `apps/finance.py` - Still calculates capital efficiency
- `apps/shelfguard_app.py` - **Only change**: Add new tabs

---

## How It Works

### User Flow:
1. **Discovery**: User searches "Starbucks" → Keepa returns 500 ASINs
2. **Pruning**: System calculates `Revenue Proxy = Units × Price`, sorts by revenue, finds the smallest set that captures 90% of total revenue
3. **Visualization**: Display pruned set as charts (market share donut, price vs BSR scatter)
4. **Pin to State**: User clicks "Create Project" → Selects mission profile (Bodyguard/Scout/Surgeon)
5. **Backfill**: Background thread fetches 90 days of Price & BSR history from Keepa
6. **Monitoring**: System generates proactive alerts based on mission profile priorities

### Data Flow:
```
Keepa API
   ↓
src/discovery.py (search + prune)
   ↓
apps/search_to_state_ui.py (visualize)
   ↓
src/persistence.py (create project in Supabase)
   ↓
src/backfill.py (fetch 90-day history)
   ↓
historical_metrics table
   ↓
src/recommendations.py (generate alerts)
   ↓
apps/search_to_state_ui.py (display resolution cards)
```

---

## Migration Strategy

### Phase 1: Parallel Systems (Current)
- Existing users: Continue using `keepa_weekly_rows` table
- New users: Use Search-to-State (project-scoped data)
- Both systems run side-by-side

### Phase 2: Hybrid Mode (Future)
- Allow existing users to "import" their current ASINs into a project
- Script to migrate `keepa_weekly_rows` → `historical_metrics`

### Phase 3: Full Migration (Long-term)
- Deprecate `keepa_weekly_rows` table
- All users on project-based model
- Enable multi-project support per user

---

## Troubleshooting

### Issue: "No products found"
**Cause**: Keepa API returned empty results

**Solutions**:
1. Check Keepa API key is valid (`KEEPA_API_KEY` in .env)
2. Try a more specific query (e.g., "Keurig K-Cups" instead of "Coffee")
3. Verify Keepa token balance

### Issue: "Backfill failed"
**Cause**: Keepa API timeout or rate limit

**Solutions**:
1. Check console for error messages
2. Reduce ASIN count (try with 50 ASINs first)
3. Set `run_async=False` in `execute_backfill()` for debugging

### Issue: "Project not appearing"
**Cause**: RLS policy blocking access

**Solutions**:
1. Verify RLS policies are active in Supabase
2. Check `user_id` is NULL (for anonymous mode)
3. Query `projects` table directly to verify insertion:
   ```sql
   SELECT * FROM projects ORDER BY created_at DESC LIMIT 5;
   ```

### Issue: "Charts not loading"
**Cause**: Backfill still in progress

**Solutions**:
1. Wait 30-60 seconds for backfill to complete
2. Check `historical_metrics` table for data:
   ```sql
   SELECT COUNT(*) FROM historical_metrics WHERE project_id = 'your-project-id';
   ```
3. Expected count: ~90 records per ASIN (1 per day × 90 days)

---

## Performance Optimization

### API Token Conservation
- **Search queries**: Cached for 24 hours (`@st.cache_data(ttl=86400)`)
- **Benefit**: 100 users searching "Starbucks" = 1 API call (not 100)

### Database Optimization
- **Indexes**: Already created on `(project_id, asin, datetime)` for fast time-series queries
- **Batch inserts**: 500 records per chunk to avoid timeouts
- **RLS**: User-scoped data prevents cross-contamination

### UI Responsiveness
- **Background backfill**: Runs in separate thread (non-blocking)
- **Lazy loading**: Resolution cards only generated on-demand
- **Plotly caching**: Charts cached in browser

---

## Next Steps

1. **Add Authentication**:
   ```python
   # In src/persistence.py:
   user_id = st.session_state.get("user", {}).get("id")

   # In pin_to_state():
   pin_to_state(..., user_id=user_id)
   ```

2. **Enable Weekly Refresh**:
   ```python
   # Create cron job to update historical_metrics weekly
   # Similar to pipelines/weekly_refresh_allscrapers.py
   ```

3. **Add Email Alerts**:
   ```python
   # Use Supabase Edge Functions to send emails for high-severity alerts
   # Trigger: INSERT on resolution_cards WHERE severity = 'high'
   ```

4. **Multi-Project Dashboard**:
   ```python
   # Allow users to view all projects in a grid layout
   # Show aggregate metrics across portfolio
   ```

---

## FAQ

**Q: Can I still use the existing dashboard?**
A: Yes! All existing functionality remains unchanged. Search-to-State is an additive feature.

**Q: How much does this cost in Keepa tokens?**
A: ~1,100 tokens per project creation (500 for search + 600 for 90-day backfill on 100 ASINs). With caching, actual cost is much lower for repeated searches.

**Q: Can I track non-Amazon marketplaces?**
A: Yes! Change `domain` parameter in `search_keepa_market()`:
- 1 = US (Amazon.com)
- 2 = UK (Amazon.co.uk)
- 3 = Germany (Amazon.de)
- etc.

**Q: What if I want to track 1,000+ ASINs?**
A: Increase the `limit` parameter in `execute_market_discovery()`. Note: Keepa charges more for larger result sets.

**Q: How do I delete a project?**
A: Currently, use Supabase directly:
```sql
DELETE FROM projects WHERE id = 'your-project-id';
-- CASCADE will auto-delete tracked_asins and historical_metrics
```

---

## Support

- **Documentation**: See `SEARCH_TO_STATE_README.md` for detailed API docs
- **Code Examples**: Check `apps/search_to_state_ui.py` for UI patterns
- **Database Schema**: Reference `schemas/search_to_state.sql`

---

**Built with Claude Code** 🛡️
