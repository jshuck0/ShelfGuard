# Database Migration Scripts

## Quick Start

### Option 1: Run Migration Helper (Recommended)

**Windows:**
```bash
cd scripts
run_migrations.bat
```

**Mac/Linux:**
```bash
cd scripts
chmod +x run_migrations.sh
./run_migrations.sh
```

This will:
- ✅ Verify your Supabase connection
- ✅ Show you what tables will be created
- ✅ Give you instructions to run SQL in Supabase
- ✅ (Optional) Copy SQL to clipboard

---

### Option 2: Manual SQL Execution (Fastest)

1. **Go to Supabase SQL Editor:**
   - Open: https://app.supabase.com/project/YOUR_PROJECT/sql/new

2. **Run Strategic Insights Schema:**
   - Copy contents of: `schemas/strategic_insights.sql`
   - Paste into SQL Editor
   - Click **Run** button
   - Wait for "Success" message

3. **Run Network Intelligence Schema:**
   - Copy contents of: `schemas/network_intelligence.sql`
   - Paste into SQL Editor
   - Click **Run** button
   - Wait for "Success" message

4. **Verify Tables Created:**
   ```sql
   SELECT table_name
   FROM information_schema.tables
   WHERE table_schema = 'public'
   ORDER BY table_name;
   ```

   Expected tables:
   - `brand_intelligence`
   - `category_intelligence`
   - `insight_outcomes`
   - `market_patterns`
   - `product_snapshots` (already exists)
   - `strategic_insights`
   - `trigger_events`

---

## Troubleshooting

### Error: "Missing Supabase credentials"

**Fix:** Set environment variables:

**Windows (PowerShell):**
```powershell
$env:SUPABASE_URL = "https://your-project.supabase.co"
$env:SUPABASE_SERVICE_KEY = "your-service-key"
```

**Mac/Linux:**
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_KEY="your-service-key"
```

**Or create `.env` file in project root:**
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
```

### Error: "Table already exists"

This is OK - it means you already ran the migration. Supabase will skip creating duplicate tables.

### Error: "Permission denied"

**Mac/Linux only:**
```bash
chmod +x run_migrations.sh
```

---

## What Gets Created

### Strategic Insights Schema (7 objects)

**Tables:**
1. `strategic_insights` - Main insights storage (recommendations, triggers, financial impact)
2. `trigger_events` - Causal market events
3. `insight_outcomes` - Prediction accuracy tracking

**Views:**
- `active_insights` - Filtered active insights
- `critical_alerts` - Critical priority only
- `opportunities` - Opportunity priority only

**Functions:**
- `expire_old_insights()` - Auto-expire old insights (30 days)

### Network Intelligence Schema (10+ objects)

**Tables:**
1. `category_intelligence` - Category benchmarks (median price, reviews, BSR)
2. `brand_intelligence` - Brand-level aggregates
3. `market_patterns` - Historical pattern library

**Extended:**
- `product_snapshots` - Added category metadata columns

**Views:**
- `latest_category_intelligence` - Most recent benchmarks
- `top_brands` - Top brands by revenue
- `reliable_patterns` - High-confidence patterns

**Functions:**
- `update_brand_intelligence()` - Auto-update brand metrics
- `calculate_category_intelligence()` - Calculate category benchmarks

---

## Next Steps After Migration

1. ✅ Verify tables created (SQL query above)
2. ✅ Tell Claude: "migrations done"
3. ✅ Claude will implement remaining code
4. ✅ Test by creating a project in Market Discovery

---

## Files in This Directory

- `run_migrations.py` - Python migration helper script
- `run_migrations.bat` - Windows batch script
- `run_migrations.sh` - Unix/Mac shell script
- `README.md` - This file

---

## Need Help?

- **Migration Issues:** Check `docs/IMPLEMENTATION_PROGRESS.md`
- **Architecture Questions:** Check `docs/MASTER_ARCHITECTURE_OVERVIEW.md`
- **Schema Details:** Read the SQL files directly (heavily commented)
