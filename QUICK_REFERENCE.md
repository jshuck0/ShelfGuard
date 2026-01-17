# ShelfGuard Search-to-State Quick Reference

## 🚀 5-Minute Setup

```bash
# 1. Install dependencies
pip install streamlit supabase pandas numpy plotly requests

# 2. Run database schema
# Copy schemas/search_to_state.sql → Supabase SQL Editor → Execute

# 3. Add to shelfguard_app.py
from apps.search_to_state_ui import render_discovery_ui

# In tabs section:
tab3 = st.tab("🔍 Market Discovery")
with tab3:
    render_discovery_ui()

# 4. Run
streamlit run apps/shelfguard_app.py
```

---

## 📚 Core API Cheat Sheet

### Discovery
```python
from src.discovery import execute_market_discovery

# Search & prune to 90%
snapshot, stats = execute_market_discovery(
    query="Starbucks",
    search_type="brand",  # or "category"
    limit=500
)

# Returns:
# - snapshot: DataFrame [asin, title, price, monthly_units, revenue_proxy, bsr]
# - stats: Dict {total_asins, pruned_asins, revenue_captured_pct}
```

### Persistence
```python
from src.persistence import pin_to_state

# Create project
project_id = pin_to_state(
    asins=snapshot["asin"].tolist(),
    project_name="My Project",
    mission_type="bodyguard",  # or "scout", "surgeon"
    user_id=None,  # or auth.uid()
    metadata=stats
)
```

### Backfill
```python
from src.backfill import execute_backfill

# Fetch 90-day history (async)
execute_backfill(
    project_id=project_id,
    asins=snapshot["asin"].tolist(),
    run_async=True  # Non-blocking
)
```

### Alerts
```python
from src.recommendations import generate_resolution_cards

# Generate alerts
alerts = generate_resolution_cards(
    df_metrics=historical_data,  # From historical_metrics table
    df_current=current_snapshot,
    mission_type="bodyguard"
)

# Returns: List[Dict] with {type, severity, asin, title, message, action}
```

---

## 🗄️ Database Quick Reference

### Tables

```sql
-- Projects
SELECT * FROM projects WHERE user_id = 'abc123';

-- Tracked ASINs
SELECT asin FROM tracked_asins WHERE project_id = 'xyz789';

-- Historical data
SELECT datetime, sales_rank, buy_box_price
FROM historical_metrics
WHERE project_id = 'xyz789' AND asin = 'B07GMLSQG5'
ORDER BY datetime DESC
LIMIT 90;

-- Alerts
SELECT * FROM resolution_cards
WHERE project_id = 'xyz789' AND is_dismissed = false
ORDER BY priority_score DESC;
```

### Useful Queries

```sql
-- Count historical records per ASIN
SELECT asin, COUNT(*) as records
FROM historical_metrics
WHERE project_id = 'xyz789'
GROUP BY asin;

-- Average BSR trend
SELECT DATE(datetime) as date, AVG(sales_rank) as avg_bsr
FROM historical_metrics
WHERE project_id = 'xyz789'
GROUP BY DATE(datetime)
ORDER BY date DESC;

-- Delete project (cascades to all related tables)
DELETE FROM projects WHERE id = 'xyz789';
```

---

## 🎨 UI Component Reference

```python
from apps.search_to_state_ui import (
    render_discovery_ui,        # Full search → prune → visualize flow
    render_pin_to_state_ui,     # Project creation UI
    render_project_dashboard,   # Alerts + historical charts
    render_project_selector,    # Dropdown for existing projects
    render_resolution_card      # Single alert card
)
```

---

## 🔧 Configuration

### Mission Profiles

| Profile | Focus | Top Priorities |
|---------|-------|----------------|
| **Bodyguard** 🛡️ | Defensive | Price undercut (1.0), Buy Box loss (1.0), Competitor surge (0.8) |
| **Scout** 🔍 | Offensive | New entrants (1.0), Rising stars (1.0), Competitor surge (0.8) |
| **Surgeon** 🔬 | Efficiency | Review gaps (1.0), Ad waste (1.0), Pricing inefficiency (0.8) |

### Alert Thresholds

```python
# Volume Stealer
bsr_improvement > 20%  # Rank improved
price_drop < -10%      # Price decreased

# Efficiency Gap
reviews < 50% of category_avg

# New Entrant
first_seen_date < 7 days ago
bsr < 50,000
```

---

## 📊 Performance Tips

### API Token Conservation
```python
# Searches are cached 24h
@st.cache_data(ttl=86400)
def search_keepa_market(...):
    # Multiple users searching "Starbucks" = 1 API call
```

### Database Optimization
```python
# Use indexes for fast queries
CREATE INDEX idx_historical_metrics_project_asin
ON historical_metrics(project_id, asin, datetime DESC);
```

### UI Responsiveness
```python
# Run backfill in background
execute_backfill(..., run_async=True)

# User doesn't wait for 30-60 second Keepa fetch
```

---

## 🐛 Common Issues

### "No products found"
```python
# Try more specific query
execute_market_discovery("Keurig K-Cups", "brand")  # ✅
execute_market_discovery("Coffee", "brand")         # ❌ Too broad
```

### "Backfill failed"
```python
# Debug with sync mode
execute_backfill(..., run_async=False)  # See error messages
```

### "Charts not loading"
```python
# Check if backfill completed
SELECT COUNT(*) FROM historical_metrics WHERE project_id = 'xyz';
# Expected: ~90 records per ASIN
```

### "Project not appearing"
```python
# Verify RLS policy
SELECT * FROM projects;  # As service_role (bypass RLS)
```

---

## 🔗 File Locations

```
src/
├── discovery.py         → Phase 1: Search & Prune
├── persistence.py       → Phase 2: Projects & RLS
├── backfill.py          → Phase 3: Historical Data
└── recommendations.py   → Phase 4: Alerts

apps/
└── search_to_state_ui.py  → Streamlit Components

schemas/
└── search_to_state.sql    → Database DDL

docs/
├── SEARCH_TO_STATE_README.md    → Full Documentation
├── INTEGRATION_GUIDE.md         → Setup Instructions
└── IMPLEMENTATION_SUMMARY.md    → Deliverables Overview
```

---

## 🎯 Workflow Example

```python
# 1. User searches market
snapshot, stats = execute_market_discovery("Starbucks", "brand")

# 2. User creates project
project_id = pin_to_state(
    asins=snapshot["asin"].tolist(),
    project_name="Starbucks K-Cups Q1",
    mission_type="scout"  # Offensive focus
)

# 3. System fetches history (background)
execute_backfill(project_id, snapshot["asin"].tolist())

# 4. Wait 30-60 seconds...

# 5. View dashboard
df_metrics = fetch_from_db(project_id)  # historical_metrics
alerts = generate_resolution_cards(
    df_metrics=df_metrics,
    df_current=current_data,
    mission_type="scout"
)

# 6. Display alerts
for alert in alerts[:5]:
    render_resolution_card(alert)
```

---

## 📞 Support

- **Setup Issues**: See `INTEGRATION_GUIDE.md`
- **API Questions**: See `SEARCH_TO_STATE_README.md`
- **Code Examples**: Check `apps/search_to_state_ui.py`

---

**Last Updated**: 2026-01-17
**Version**: 1.0.0
**Status**: Production Ready ✅
