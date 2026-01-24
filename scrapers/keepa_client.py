import pandas as pd
import numpy as np
import os
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- CONSTANTS (Preserved) ---
IDX_AMAZON = 0
IDX_NEW = 1
IDX_USED = 2
IDX_SALES_RANK = 3
IDX_LIST_PRICE = 4
IDX_NEW_FBA = 10
IDX_NEW_FBM = 7 
IDX_BUY_BOX = 18
IDX_COUNT_NEW = 11
IDX_COUNT_USED = 12
IDX_RATING = 16          # Keepa Python lib: csv[16] = rating (stored as rating*10)
IDX_COUNT_REVIEWS = 17   # Keepa Python lib: csv[17] = review count

SERIES_MAP = {
    IDX_AMAZON: ("amazon_price", "price"),
    IDX_NEW: ("new_price", "price"),
    IDX_NEW_FBA: ("new_fba_price", "price"),
    IDX_SALES_RANK: ("sales_rank", "rank"),
    IDX_BUY_BOX: ("buy_box_price", "price"),
    IDX_COUNT_NEW: ("new_offer_count", "count"),
    IDX_COUNT_USED: ("used_offer_count", "count"),
    IDX_RATING: ("rating", "rating"),
    IDX_COUNT_REVIEWS: ("review_count", "count")
}

MAX_RANK_GAP_WEEKS = 3
MAX_PRICE_FFILL_WEEKS = 4
AMAZON_RETAIL_ID = "ATVPDKIKX0DER"

# --- HELPER FUNCTIONS (Preserved) ---

def to_week_start(dt):
    """Aligns date to Monday of that week."""
    if pd.isna(dt): return dt
    ts = pd.to_datetime(dt)
    if ts.tz is not None: ts = ts.tz_localize(None)
    return (ts - pd.to_timedelta(ts.weekday(), unit='D')).replace(hour=0, minute=0, second=0, microsecond=0)

def rolling_36m_start(today: date | None = None) -> date:
    today = today or date.today()
    return date(today.year, today.month, 1) - relativedelta(months=36)

def _keepa_time_to_dt(keepa_minutes):
    """Converts Keepa Minutes (int) to Datetime (Vectorized)."""
    base = np.datetime64('2011-01-01')
    return base + np.array(keepa_minutes, dtype='timedelta64[m]')

def _to_datetime_resilient(x):
    """Handles various date formats safely."""
    if x is None: return pd.to_datetime([])
    dt = pd.to_datetime(x, errors='coerce')
    if dt.isna().all():
        x_num = pd.to_numeric(x, errors='coerce')
        if not np.isnan(x_num).all():
            dt = pd.Timestamp("2011-01-01") + pd.to_timedelta(x_num, unit="m")
    return dt.tz_localize(None) if hasattr(dt, 'tz_localize') else dt

import re

def _extract_attributes(product):
    """
    Booth-Ready Logic: Parses flavor and count from the title.
    Fallback to ASIN if Title is missing from the current JSON batch.
    """
    title = product.get("title")
    asin = product.get("asin", "Unknown ASIN")
    
    # If title is missing, we can't parse flavor, so return the ASIN
    if not title or str(title) == "None":
        return f"Product {asin}"

    # 1. Try to find the 'Count' (e.g., '80 ct' or '24 Count')
    count_match = re.search(r'(\d+)\s?(ct|Count|Pods|pk|Pack)', title, re.IGNORECASE)
    count_str = count_match.group(0) if count_match else ""

    # 2. Try to identify the Roast/Flavor
    flavors = ["Pike Place", "French Roast", "Veranda", "Caffe Verona", "Sumatra", "Caramel", "Breakfast Blend"]
    found_flavor = "Standard"
    for f in flavors:
        if f.lower() in title.lower():
            found_flavor = f
            break
            
    # 3. Combine them
    if count_str:
        return f"{found_flavor} | {count_str}"
    return found_flavor

def _extract_fba_fees(product):
    """Robustly finds FBA Fees (cents to dollars)."""
    fba_obj = product.get("fbaFees")
    if fba_obj and isinstance(fba_obj, dict):
        fee = fba_obj.get("pickAndPackFee")
        if fee: return float(fee) / 100.0
    try:
        fees = product.get("stats", {}).get("current", {}).get("fbaFees")
        if fees: return float(fees) / 100.0
    except:
        pass
    return None

def parse_buybox_stats(product):
    """Parses 'buyBoxSellerIdHistory' to calculate Amazon's share of the Buy Box."""
    hist = product.get("buyBoxSellerIdHistory") or []
    if len(hist) < 2: return pd.DataFrame()
    
    df = pd.DataFrame({"datetime": _to_datetime_resilient(hist[::2]), "seller_id": hist[1::2]})
    df = df.sort_values("datetime")
    df["next_dt"] = df["datetime"].shift(-1).fillna(pd.Timestamp.now())
    df["duration"] = (df["next_dt"] - df["datetime"]).dt.total_seconds()
    df["week_start"] = df["datetime"].apply(to_week_start)
    df["is_switch"] = df["seller_id"].ne(df["seller_id"].shift()) & df["seller_id"].shift().notna()
    
    weekly = df.groupby("week_start").agg(
        total_sec=("duration", "sum"),
        amz_sec=("duration", lambda x: df.loc[x.index, "duration"][df.loc[x.index, "seller_id"] == AMAZON_RETAIL_ID].sum()),
        buy_box_switches=("is_switch", "sum")
    ).reset_index()
    
    weekly["amazon_bb_share"] = (weekly["amz_sec"] / weekly["total_sec"]).fillna(0)
    return weekly[["week_start", "amazon_bb_share", "buy_box_switches"]]

# --- EXTRACTION ENGINE ---

def extract_weekly_facts(product, window_start=None):
    """Process a single product. Handles RAW JSON (Integer Keys)."""
    asin = product.get("asin", "UNK")
    p_asin = product.get("parentAsin", asin)
    title = product.get("title", "Unknown")
    brand = product.get("brand", "Unknown")
    manufacturer = product.get("manufacturer", brand)
    # NEW CODE
    images_raw = product.get("imagesCSV")
    if images_raw and isinstance(images_raw, str):
        images = images_raw.split(",")
        main_image = f"https://m.media-amazon.com/images/I/{images[0]}" if images else None
    else:
        main_image = None
    var_attrs = _extract_attributes(product)
    
    fba_fee = _extract_fba_fees(product)
    pkg_weight = product.get("packageWeight", 0)
    dims = [product.get("packageHeight", 0), product.get("packageLength", 0), product.get("packageWidth", 0)]
    pkg_vol_cf = np.prod(dims) * 3.53147e-8 if all(dims) else 0.05
    
    # === NEW CRITICAL METRICS (2026-01-21) ===
    # Amazon's actual monthly sold estimate (much better than BSR formula!)
    monthly_sold = product.get("monthlySold", 0) or 0
    
    # Pack size for per-unit normalization
    number_of_items = product.get("numberOfItems", 1) or 1
    
    # Buy Box ownership flags
    buybox_is_amazon = product.get("buyBoxIsAmazon", None)
    buybox_is_fba = product.get("buyBoxIsFBA", None)
    buybox_is_backorder = product.get("buyBoxIsBackorder", False) or False
    buybox_is_unqualified = product.get("buyBoxIsUnqualified", False) or False
    
    # Seller IDs array - for true competitor count
    seller_ids_raw = product.get("sellerIds", []) or []
    seller_count = len(seller_ids_raw) if seller_ids_raw else 0
    has_amazon_seller = AMAZON_RETAIL_ID in seller_ids_raw if seller_ids_raw else False
    
    # Stats object for pre-calculated metrics
    stats = product.get("stats", {}) or {}
    
    # Helper to safely extract scalar from potentially list values
    def _safe_stat(val, default=0):
        """Extract scalar from Keepa stat that might be a list."""
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return val[-1] if len(val) > 0 and val[-1] is not None else default
        if isinstance(val, (int, float)):
            return val
        return default
    
    # OOS counts (more useful than just percentage)
    oos_count_amazon_30 = _safe_stat(stats.get("outOfStockCountAmazon30"), 0)
    oos_count_amazon_90 = _safe_stat(stats.get("outOfStockCountAmazon90"), 0)
    oos_pct_30 = _safe_stat(stats.get("outOfStockPercentage30"), 0)
    oos_pct_90 = _safe_stat(stats.get("outOfStockPercentage90"), 0)
    
    # Buy Box stats from Keepa (backup to our calculation)
    bb_stats_amazon_30 = _safe_stat(stats.get("buyBoxStatsAmazon30"), None)
    bb_stats_amazon_90 = _safe_stat(stats.get("buyBoxStatsAmazon90"), None)
    bb_stats_top_seller_30 = _safe_stat(stats.get("buyBoxStatsTopSeller30"), None)
    bb_stats_seller_count_30 = _safe_stat(stats.get("buyBoxStatsSellerCount30"), None)
    
    # Velocity deltas (pre-calculated by Keepa)
    delta_pct_30 = stats.get("deltaPercent30", []) or []
    delta_pct_90 = stats.get("deltaPercent90", []) or []
    # BSR is at index 3 - safely extract
    try:
        velocity_30d = float(delta_pct_30[3]) if delta_pct_30 and len(delta_pct_30) > 3 and delta_pct_30[3] is not None else None
    except (TypeError, ValueError, IndexError):
        velocity_30d = None
    try:
        velocity_90d = float(delta_pct_90[3]) if delta_pct_90 and len(delta_pct_90) > 3 and delta_pct_90[3] is not None else None
    except (TypeError, ValueError, IndexError):
        velocity_90d = None
    
    # Subscribe & Save eligibility
    is_sns = product.get("isSNS", False) or False

    csv_data = product.get("csv", {})
    limit_ts = to_week_start(pd.Timestamp(window_start or "2023-01-01"))
    frames = []

    for keepa_idx, (col_name, _) in SERIES_MAP.items():
        try:
            raw_list = csv_data[keepa_idx] if isinstance(csv_data, list) else csv_data.get(str(keepa_idx)) or csv_data.get(keepa_idx)
        except (IndexError, KeyError, TypeError):
            continue
            
        if not raw_list or len(raw_list) == 0: continue
        if len(raw_list) % 2 != 0: raw_list = raw_list[:-1]

        arr = np.array(raw_list, dtype=float).reshape(-1, 2)
        times = _keepa_time_to_dt(arr[:, 0])
        vals = arr[:, 1]

        # CRITICAL FIX: Replace Keepa special values (-1 = no data, -2 = out of stock)
        # BEFORE any normalization. Otherwise -1 becomes -0.01 after cents→dollars conversion.
        vals = np.where(np.isin(vals, [-1, -2]), np.nan, vals)

        # --- Normalization (Cents to Dollars) ---
        if "price" in col_name:
            vals = vals / 100.0

        v = pd.Series(vals, index=times).dropna()  # NaN already set above
        if v.empty: continue
            
        tmp = pd.DataFrame({"datetime": v.index, col_name: v.values})
        tmp["week_start"] = tmp["datetime"].apply(to_week_start)
        agg_func = "mean" if "rank" in col_name else "last"
        weekly = tmp[tmp["week_start"] >= limit_ts].groupby("week_start", as_index=False)[col_name].agg(agg_func)
        
        if not weekly.empty: 
            frames.append(weekly)

    bb = parse_buybox_stats(product)
    if not bb.empty:
        bb = bb[bb["week_start"] >= limit_ts]
        frames.append(bb)

    if not frames: return pd.DataFrame()
    
    out = frames[0]
    for f in frames[1:]: out = out.merge(f, on="week_start", how="outer")
    
    grid_end = to_week_start(pd.Timestamp.now())
    all_weeks = pd.date_range(start=limit_ts, end=grid_end, freq="7D")
    out = out.set_index("week_start").reindex(all_weeks).rename_axis("week_start").reset_index()

    # Core metadata (inserted at start)
    meta = [
        ("asin", asin), ("parent_asin", p_asin), ("title", title), 
        ("brand", brand), ("manufacturer", manufacturer), 
        ("main_image", main_image), ("variation_attributes", var_attrs),
        ("fba_fees", fba_fee), ("package_weight_g", pkg_weight), 
        ("package_vol_cf", pkg_vol_cf)
    ]
    for c, val in meta: out.insert(0, c, val)
    
    # === NEW CRITICAL METRICS - Added as columns (2026-01-21) ===
    # These are product-level (not time-series), so same value for all weeks
    out["monthly_sold"] = monthly_sold
    out["number_of_items"] = number_of_items
    out["buybox_is_amazon"] = buybox_is_amazon
    out["buybox_is_fba"] = buybox_is_fba
    out["buybox_is_backorder"] = buybox_is_backorder
    out["buybox_is_unqualified"] = buybox_is_unqualified
    out["seller_count"] = seller_count
    out["has_amazon_seller"] = has_amazon_seller
    out["oos_count_amazon_30"] = oos_count_amazon_30
    out["oos_count_amazon_90"] = oos_count_amazon_90
    # Normalize percentages (Keepa stats are 0-100 integers)
    # Check >= 1.0 to verify if it's 0-100 scale.
    # Note: 1.0 (100% on 0-1 scale) becomes 0.01 (1%) with this check, but Keepa returns INTs so 100% is 100.
    # The only ambiguity is 1 (1% vs 100%). Given Keepa returns ints 0-100, 1 is almost certainly 1%.
    # So we assume anything >= 1 is on 0-100 scale.
    out["oos_pct_30"] = (oos_pct_30 / 100.0) if isinstance(oos_pct_30, (int, float)) and oos_pct_30 >= 1 else oos_pct_30
    out["oos_pct_90"] = (oos_pct_90 / 100.0) if isinstance(oos_pct_90, (int, float)) and oos_pct_90 >= 1 else oos_pct_90
    out["bb_stats_amazon_30"] = (bb_stats_amazon_30 / 100.0) if isinstance(bb_stats_amazon_30, (int, float)) and bb_stats_amazon_30 >= 1 else bb_stats_amazon_30
    out["bb_stats_amazon_90"] = (bb_stats_amazon_90 / 100.0) if isinstance(bb_stats_amazon_90, (int, float)) and bb_stats_amazon_90 >= 1 else bb_stats_amazon_90
    out["bb_stats_top_seller_30"] = (bb_stats_top_seller_30 / 100.0) if isinstance(bb_stats_top_seller_30, (int, float)) and bb_stats_top_seller_30 >= 1 else bb_stats_top_seller_30
    out["bb_stats_seller_count_30"] = bb_stats_seller_count_30
    out["velocity_30d"] = velocity_30d
    out["velocity_90d"] = velocity_90d
    out["is_sns"] = is_sns
    
    return out

def build_keepa_weekly_table(products, window_start=None):
    all_dfs = [extract_weekly_facts(p, window_start) for p in products if p]
    # Filter out empty DataFrames
    all_dfs = [df for df in all_dfs if not df.empty]
    if not all_dfs: return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Check if we have any data
    if df.empty or 'asin' not in df.columns:
        return pd.DataFrame()
    
    # --- FIX 2: Deduplication by ASIN and Week ---
    df = df.drop_duplicates(subset=["asin", "week_start"])
    
    # --- FIX: Scalarize columns that might contain list values ---
    # Some Keepa fields can come back as lists, which cause comparison errors
    def _scalarize_column(series):
        """Convert any list values in a series to scalars."""
        def _to_scalar(val):
            if val is None:
                return np.nan
            if isinstance(val, (list, tuple)):
                return val[-1] if len(val) > 0 else np.nan
            return val
        return series.apply(_to_scalar)
    
    # Scalarize numeric columns that might have list values
    numeric_cols = ['sales_rank', 'monthly_sold', 'number_of_items', 'seller_count',
                    'oos_count_amazon_30', 'oos_count_amazon_90', 'oos_pct_30', 'oos_pct_90',
                    'bb_stats_amazon_30', 'bb_stats_amazon_90', 'bb_stats_top_seller_30',
                    'bb_stats_seller_count_30', 'velocity_30d', 'velocity_90d']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _scalarize_column(df[col])
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Build effective price from available price columns
    # Priority: buy_box > amazon > new > new_fba
    eff_p = pd.Series(np.nan, index=df.index)
    for col in ["buy_box_price", "amazon_price", "new_price", "new_fba_price"]:
        if col in df.columns:
            eff_p = eff_p.fillna(df[col])
    df["eff_p"] = eff_p

    # Fill price gaps using forward-fill, then backward-fill for leading gaps
    # This ensures we use the most recent known price, but also fill early weeks
    # that had no price data by using the earliest available price
    # FIX: Handle pandas 2.0+ where ffill/bfill might fail on edge cases
    def safe_fill_price(x):
        if x.isna().all():
            return x
        try:
            return x.ffill(limit=MAX_PRICE_FFILL_WEEKS).bfill(limit=MAX_PRICE_FFILL_WEEKS)
        except ValueError:
            return x
    
    df["filled_price"] = df.groupby("asin")["eff_p"].transform(safe_fill_price)

    # ULTIMATE FALLBACK: For any remaining NaN prices, use the ASIN's mean price
    # This handles cases where price data gaps exceed MAX_PRICE_FFILL_WEEKS
    asin_mean_prices = df.groupby("asin")["eff_p"].transform("mean")
    df["filled_price"] = df["filled_price"].fillna(asin_mean_prices)

    # Fill BSR with interpolation if available
    if "sales_rank" in df.columns:
        # Interpolate BSR to handle gaps smoothly
        # FIX: Handle pandas 2.0+ where interpolate on all-NaN raises ValueError
        def safe_interpolate_bsr(x):
            if len(x) <= 1 or x.isna().all():
                return x.ffill().bfill()
            try:
                return x.interpolate(limit=MAX_RANK_GAP_WEEKS).ffill().bfill()
            except ValueError:
                return x.ffill().bfill()
        
        df["sales_rank_filled"] = df.groupby("asin")["sales_rank"].transform(safe_interpolate_bsr)
    else:
        # No BSR data - use high default (low confidence)
        df["sales_rank_filled"] = 100000

    # Fill any remaining NaN BSRs with ASIN's mean
    asin_mean_bsr = df.groupby("asin")["sales_rank_filled"].transform("mean")
    df["sales_rank_filled"] = df["sales_rank_filled"].fillna(asin_mean_bsr).fillna(100000)

    # === UNITS ESTIMATION: Prefer Amazon's monthlySold over BSR formula ===
    # monthlySold is Amazon's actual estimate (when available) - much more accurate!
    # Fallback to BSR formula when monthlySold is not available
    
    # BSR-based estimate (CALIBRATED FOR GROCERY VELOCITY)
    bsr_monthly_units = (145000.0 * (df["sales_rank_filled"].clip(lower=1) ** -0.9))
    
    # Use monthlySold if available and > 0, otherwise use BSR formula
    if "monthly_sold" in df.columns:
        # Prefer Amazon's estimate, fallback to BSR formula
        df["monthly_units"] = df["monthly_sold"].where(
            df["monthly_sold"] > 0, 
            bsr_monthly_units
        )
        df["units_source"] = np.where(df["monthly_sold"] > 0, "amazon_monthly_sold", "bsr_formula")
    else:
        df["monthly_units"] = bsr_monthly_units
        df["units_source"] = "bsr_formula"

    # SCALE TO WEEKLY BUCKET
    df["estimated_units"] = df["monthly_units"] * (7 / 30)
    df["weekly_sales_filled"] = df["estimated_units"] * df["filled_price"].fillna(0)
    
    # Per-unit price (for fair comparison across pack sizes)
    if "number_of_items" in df.columns:
        df["price_per_unit"] = df["filled_price"] / df["number_of_items"].clip(lower=1)
    else:
        df["price_per_unit"] = df["filled_price"]
    
    # === DATA HEALER INTEGRATION ===
    # Apply comprehensive gap-filling for ALL metrics before data reaches AI
    # This ensures review_count, rating, amazon_bb_share, etc. are filled
    try:
        from utils.data_healer import clean_and_interpolate_metrics
        df = clean_and_interpolate_metrics(df, group_by_column="asin", verbose=False)
    except ImportError:
        pass  # Data healer not available, use unhealed data
    
    return df