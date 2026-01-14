import pandas as pd
import numpy as np
import os
from datetime import date
from dateutil.relativedelta import relativedelta

# FULL COMPETITIVE SERIES MAPPING
SERIES_MAP = {
    "AMAZON":           ("amazon_price", "price"),
    "NEW":              ("new_price", "price"),
    "NEW_FBA":          ("new_fba_price", "price"),  # Added
    "NEW_FBM_SHIPPING": ("new_fbm_price", "price"),  # Added
    "LISTPRICE":        ("list_price", "price"),
    "SALES":            ("sales_rank", "rank"),
    "BUY_BOX_SHIPPING": ("buy_box_price", "price"),
    "COUNT_NEW":        ("new_offer_count", "count"),
    "COUNT_USED":       ("used_offer_count", "count"), # Added
    "RATING":           ("rating", "rating"),
    "COUNT_REVIEWS":    ("review_count", "count"),
}

MAX_RANK_GAP_WEEKS = 3
MAX_PRICE_FFILL_WEEKS = 4
AMAZON_RETAIL_ID = "ATVPDKIKX0DER"

def to_week_start(dt):
    if pd.isna(dt): return dt
    ts = pd.to_datetime(dt)
    if ts.tz is not None: ts = ts.tz_localize(None)
    return (ts - pd.to_timedelta(ts.weekday(), unit='D')).replace(hour=0, minute=0, second=0, microsecond=0)

def rolling_36m_start(today: date | None = None) -> date:
    today = today or date.today()
    return date(today.year, today.month, 1) - relativedelta(months=36)

def _to_datetime_resilient(x):
    if x is None: return pd.to_datetime([])
    dt = pd.to_datetime(x, errors='coerce')
    if dt.isna().all():
        x_num = pd.to_numeric(x, errors='coerce')
        if not np.isnan(x_num).all():
            dt = pd.Timestamp("2011-01-01") + pd.to_timedelta(x_num, unit="m")
    return dt.tz_localize(None) if hasattr(dt, 'tz_localize') else dt

def parse_buybox_stats(product):
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

def extract_weekly_facts(product, window_start=None):
    asin = product.get("asin", "UNK")
    p_asin = product.get("parentAsin", asin)
    title = product.get("title", "Unknown")
    
    # NEW STATIC METADATA
    brand = product.get("brand", "Unknown")
    manufacturer = product.get("manufacturer", brand)
    images = product.get("imagesCSV", "").split(",")
    main_image = f"https://m.media-amazon.com/images/I/{images[0]}" if images[0] else None
    
    # FBA Fee (taking the most recent pick and pack fee)
    fba_fees = product.get("fbaFees", {}) or {}
    fba_fee = fba_fees.get("pickAndPackFee", 0) / 100.0 # Convert cents to dollars

    data_tier = product.get("data", {}) or {}
    limit_ts = to_week_start(pd.Timestamp(window_start or "2023-01-01"))
    
    frames = []
    for keepa_key, (col_name, _) in SERIES_MAP.items():
        vals, times = data_tier.get(keepa_key), data_tier.get(f"{keepa_key}_time")
        if vals is not None and times is not None:
            dt = _to_datetime_resilient(times)
            v = pd.Series(vals, dtype="float64").replace([-1, -2], np.nan)
            tmp = pd.DataFrame({"datetime": dt, col_name: v}).dropna()
            if not tmp.empty:
                tmp["week_start"] = tmp["datetime"].apply(to_week_start)
                weekly = tmp[tmp["week_start"] >= limit_ts].groupby("week_start", as_index=False)[col_name].last()
                if not weekly.empty: frames.append(weekly)

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

    # Inject static metadata into every row
    for c, val in [("asin", asin), ("parent_asin", p_asin), ("title", title), 
                   ("brand", brand), ("manufacturer", manufacturer), 
                   ("main_image", main_image), ("fba_fee", fba_fee)]:
        out.insert(0, c, val)
    return out

def build_keepa_weekly_table(products, window_start=None):
    all_dfs = [extract_weekly_facts(p, window_start) for p in products if p]
    if not all_dfs: return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)
    df["eff_p"] = df.get("buy_box_price", np.nan).fillna(df.get("amazon_price", np.nan)).fillna(df.get("new_price", np.nan))
    df["filled_price"] = df.groupby("asin")["eff_p"].ffill(limit=MAX_PRICE_FFILL_WEEKS)
    if "sales_rank" in df.columns:
        df["sales_rank_filled"] = df.groupby("asin")["sales_rank"].transform(lambda x: x.interpolate(limit=MAX_RANK_GAP_WEEKS))
        df["estimated_units"] = (50000.0 / np.power(df["sales_rank_filled"].clip(lower=1), 1.05))
        df["weekly_sales_filled"] = df["estimated_units"] * df["filled_price"].fillna(0)
    return df