import pandas as pd

def to_week_start(dt):
    """
    Safely converts a datetime to the Monday of that week.
    Returns NaT if the input is null or invalid.
    """
    if dt is None or pd.isna(dt):
        return pd.NaT
    
    # Ensure it's a timestamp object
    dt = pd.to_datetime(dt)
    
    # Calculate Monday and strip the time (normalize)
    try:
        monday = dt - pd.to_timedelta(dt.weekday(), unit="D")
        return monday.normalize()
    except Exception:
        return pd.NaT
