import pandas as pd

def to_week_start(dt):
    """
    Safely converts a datetime to the Monday of that week.
    Returns NaT if the input is null or invalid.

    Performance: Optimized early returns and exception handling.
    """
    if dt is None or pd.isna(dt):
        return pd.NaT

    try:
        # Ensure it's a timestamp object (one-time conversion)
        dt = pd.to_datetime(dt)

        # Calculate Monday and normalize (combined operation)
        monday = dt - pd.to_timedelta(dt.weekday(), unit="D")
        return monday.normalize()
    except (ValueError, TypeError):
        return pd.NaT
