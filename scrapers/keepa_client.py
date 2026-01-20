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
IDX_COUNT_REVIEWS = 16
IDX_RATING = 17

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

    meta = [
        ("asin", asin), ("parent_asin", p_asin), ("title", title), 
        ("brand", brand), ("manufacturer", manufacturer), 
        ("main_image", main_image), ("variation_attributes", var_attrs),
        ("fba_fees", fba_fee), ("package_weight_g", pkg_weight), 
        ("package_vol_cf", pkg_vol_cf)
    ]
    for c, val in meta: out.insert(0, c, val)
    return out

def build_keepa_weekly_table(products, window_start=None):
    all_dfs = [extract_weekly_facts(p, window_start) for p in products if p]
    if not all_dfs: return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # --- FIX 2: Deduplication by ASIN and Week ---
    df = df.drop_duplicates(subset=["asin", "week_start"])

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
    df["filled_price"] = df.groupby("asin")["eff_p"].transform(
        lambda x: x.ffill(limit=MAX_PRICE_FFILL_WEEKS).bfill(limit=MAX_PRICE_FFILL_WEEKS)
    )

    # ULTIMATE FALLBACK: For any remaining NaN prices, use the ASIN's mean price
    # This handles cases where price data gaps exceed MAX_PRICE_FFILL_WEEKS
    asin_mean_prices = df.groupby("asin")["eff_p"].transform("mean")
    df["filled_price"] = df["filled_price"].fillna(asin_mean_prices)

    # Fill BSR with interpolation if available
    if "sales_rank" in df.columns:
        # Interpolate BSR to handle gaps smoothly
        df["sales_rank_filled"] = df.groupby("asin")["sales_rank"].transform(
            lambda x: x.interpolate(limit=MAX_RANK_GAP_WEEKS).ffill().bfill()
        )
    else:
        # No BSR data - use high default (low confidence)
        df["sales_rank_filled"] = 100000

    # Fill any remaining NaN BSRs with ASIN's mean
    asin_mean_bsr = df.groupby("asin")["sales_rank_filled"].transform("mean")
    df["sales_rank_filled"] = df["sales_rank_filled"].fillna(asin_mean_bsr).fillna(100000)

    # CALIBRATED FOR GROCERY VELOCITY
    monthly_units = (145000.0 * (df["sales_rank_filled"].clip(lower=1) ** -0.9))

    # SCALE TO WEEKLY BUCKET
    df["estimated_units"] = monthly_units * (7 / 30)
    df["weekly_sales_filled"] = df["estimated_units"] * df["filled_price"].fillna(0)
    
    # === DATA HEALER INTEGRATION ===
    # Apply comprehensive gap-filling for ALL metrics before data reaches AI
    # This ensures review_count, rating, amazon_bb_share, etc. are filled
    try:
        from utils.data_healer import clean_and_interpolate_metrics
        df = clean_and_interpolate_metrics(df, group_by_column="asin", verbose=False)
    except ImportError:
        pass  # Data healer not available, use unhealed data
    
    return df