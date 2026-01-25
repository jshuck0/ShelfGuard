"""
World Context: Calendar, Seasonality, and External Events

Injects real-world context into the AI analysis to prevent false positives.
A 10% sales drop in February might be outperformance if the category is down 15%.

Key behaviors:
1. Identify current season and its impact on the category
2. Flag upcoming holidays and events
3. Provide category benchmarks for comparison
4. Allow the AI to filter noise from signal
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from enum import Enum


class Season(Enum):
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"


# =============================================================================
# SEASONAL PATTERNS BY CATEGORY
# =============================================================================

CATEGORY_SEASONALITY = {
    # Personal care products
    "deodorant": {
        Season.WINTER: {"expected_change": -15, "note": "Low demand in winter"},
        Season.SPRING: {"expected_change": 5, "note": "Warming weather increases demand"},
        Season.SUMMER: {"expected_change": 25, "note": "Peak season for deodorant"},
        Season.FALL: {"expected_change": -5, "note": "Cooling weather reduces demand"},
    },
    "body_wash": {
        Season.WINTER: {"expected_change": 10, "note": "Dry skin drives moisturizing products"},
        Season.SPRING: {"expected_change": 0, "note": "Stable demand"},
        Season.SUMMER: {"expected_change": 15, "note": "More showers = more body wash"},
        Season.FALL: {"expected_change": 0, "note": "Stable demand"},
    },
    "sunscreen": {
        Season.WINTER: {"expected_change": -60, "note": "Minimal demand except ski/travel"},
        Season.SPRING: {"expected_change": 30, "note": "Ramp-up as weather warms"},
        Season.SUMMER: {"expected_change": 80, "note": "Peak season"},
        Season.FALL: {"expected_change": -30, "note": "Sharp decline post-Labor Day"},
    },
    # Default for unknown categories
    "default": {
        Season.WINTER: {"expected_change": 0, "note": ""},
        Season.SPRING: {"expected_change": 0, "note": ""},
        Season.SUMMER: {"expected_change": 0, "note": ""},
        Season.FALL: {"expected_change": 0, "note": ""},
    },
}


# =============================================================================
# HOLIDAYS AND EVENTS
# =============================================================================

def get_upcoming_events(current_date: date, lookahead_days: int = 30) -> List[Dict[str, Any]]:
    """Get relevant upcoming events that may impact sales."""
    
    # Define events by month-day
    events = [
        # Q1
        {"month": 1, "day": 1, "name": "New Year's Day", "impact": "Health/fitness resolutions spike"},
        {"month": 2, "day": 2, "name": "Super Bowl", "impact": "Snack focus, low personal care"},
        {"month": 2, "day": 14, "name": "Valentine's Day", "impact": "Gift sets, premium products"},
        {"month": 2, "day": 15, "name": "Presidents Day", "impact": "Sales events"},
        
        # Q2
        {"month": 3, "day": 17, "name": "St. Patrick's Day", "impact": "Minimal impact"},
        {"month": 4, "day": 1, "name": "Easter (approx)", "impact": "Gift potential, family gatherings"},
        {"month": 5, "day": 12, "name": "Mother's Day", "impact": "Gift sets, skincare, premium"},
        {"month": 5, "day": 26, "name": "Memorial Day", "impact": "Outdoor products, summer kickoff"},
        
        # Q3
        {"month": 6, "day": 15, "name": "Father's Day", "impact": "Men's grooming, gift sets"},
        {"month": 7, "day": 4, "name": "Independence Day", "impact": "Outdoor, travel sizes"},
        {"month": 7, "day": 15, "name": "Prime Day (approx)", "impact": "Massive discounting, stock up"},
        {"month": 9, "day": 1, "name": "Labor Day", "impact": "End of summer, back to school"},
        
        # Q4
        {"month": 10, "day": 31, "name": "Halloween", "impact": "Minimal for personal care"},
        {"month": 11, "day": 11, "name": "Veterans Day", "impact": "Sales events"},
        {"month": 11, "day": 28, "name": "Black Friday (approx)", "impact": "Major discounting expected"},
        {"month": 12, "day": 2, "name": "Cyber Monday (approx)", "impact": "Online sales surge"},
        {"month": 12, "day": 25, "name": "Christmas", "impact": "Gift sets, premium products"},
    ]
    
    upcoming = []
    for event in events:
        # Create this year's date
        try:
            event_date = date(current_date.year, event["month"], event["day"])
            
            # If event is in the past this year, check next year
            if event_date < current_date:
                event_date = date(current_date.year + 1, event["month"], event["day"])
            
            days_until = (event_date - current_date).days
            if 0 <= days_until <= lookahead_days:
                upcoming.append({
                    "date": event_date.isoformat(),
                    "name": event["name"],
                    "impact": event["impact"],
                    "days_until": days_until,
                })
        except ValueError:
            continue  # Skip invalid dates
    
    # Sort by days until
    upcoming.sort(key=lambda e: e["days_until"])
    return upcoming


# =============================================================================
# SEASON DETECTION
# =============================================================================

def get_current_season(current_date: date) -> Season:
    """Determine current season from date."""
    month = current_date.month
    
    if month in [12, 1, 2]:
        return Season.WINTER
    elif month in [3, 4, 5]:
        return Season.SPRING
    elif month in [6, 7, 8]:
        return Season.SUMMER
    else:
        return Season.FALL


def get_season_name(season: Season) -> str:
    """Get human-readable season name."""
    return season.value.capitalize()


# =============================================================================
# MAIN CONTEXT BUILDER
# =============================================================================

def get_world_context(
    current_date: datetime = None,
    category: str = "default",
    category_benchmarks: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Build the world context dictionary for LLM injection.
    
    Args:
        current_date: The date to analyze (defaults to now)
        category: Product category for seasonality lookup
        category_benchmarks: Optional real benchmarks from data
        
    Returns:
        Dictionary with season, events, and benchmarks
    """
    if current_date is None:
        current_date = datetime.now()
    
    current_date_obj = current_date.date() if isinstance(current_date, datetime) else current_date
    
    # Get season
    season = get_current_season(current_date_obj)
    
    # Get seasonality for this category
    category_key = category.lower().replace(" ", "_").replace("-", "_")
    seasonality = CATEGORY_SEASONALITY.get(category_key, CATEGORY_SEASONALITY["default"])
    season_data = seasonality.get(season, {"expected_change": 0, "note": ""})
    
    # Get upcoming events
    upcoming_events = get_upcoming_events(current_date_obj, lookahead_days=30)
    
    # Build context
    context = {
        # Current state
        "date": current_date_obj.isoformat(),
        "day_of_week": current_date_obj.strftime("%A"),
        "month": current_date_obj.strftime("%B"),
        "year": current_date_obj.year,
        
        # Season
        "season": get_season_name(season),
        "season_raw": season.value,
        
        # Category seasonality
        "category": category,
        "expected_seasonal_change": season_data["expected_change"],
        "seasonality_note": season_data["note"],
        
        # Upcoming events
        "upcoming_events": upcoming_events,
        
        # Economic context (placeholder - could be enhanced with real data)
        "economic_climate": "Stable",
        "consumer_sentiment": "Cautious spending",
        
        # Category benchmarks (from data if available)
        "category_benchmarks": category_benchmarks or {},
    }
    
    return context


def format_world_context_for_llm(context: Dict[str, Any]) -> str:
    """
    Format world context as a string for LLM prompt injection.
    """
    lines = []
    lines.append("=== WORLD CONTEXT ===")
    lines.append(f"Date: {context['date']} ({context['day_of_week']})")
    lines.append(f"Season: {context['season']}")
    lines.append("")
    
    # Seasonality
    if context.get("seasonality_note"):
        lines.append(f"Category Seasonality: {context['seasonality_note']}")
        expected = context.get("expected_seasonal_change", 0)
        if expected != 0:
            lines.append(f"Expected seasonal change: {expected:+d}% vs annual average")
        lines.append("")
    
    # Upcoming events
    events = context.get("upcoming_events", [])
    if events:
        lines.append("Upcoming Events:")
        for event in events[:3]:  # Max 3 events
            lines.append(f"  - {event['name']} in {event['days_until']} days: {event['impact']}")
        lines.append("")
    
    # Economic context
    lines.append(f"Economic Climate: {context.get('economic_climate', 'Unknown')}")
    lines.append(f"Consumer Sentiment: {context.get('consumer_sentiment', 'Unknown')}")
    lines.append("")
    
    # Instructions
    lines.append("INSTRUCTION: Use this context to filter false positives.")
    lines.append("A sales drop during low season may be OUTPERFORMANCE if category is down more.")
    
    return "\n".join(lines)


# =============================================================================
# GOOGLE TRENDS INTEGRATION (for search volume context)
# =============================================================================

def get_search_trends(keywords: List[str], timeframe: str = "today 3-m") -> Optional[Dict]:
    """
    Pull relative search interest from Google Trends.
    
    Requires: pip install pytrends
    
    Args:
        keywords: List of keywords to check (max 5)
        timeframe: Google Trends timeframe string
        
    Returns:
        Dictionary with trend data or None if unavailable
    """
    try:
        from pytrends.request import TrendReq
        
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords[:5], timeframe=timeframe, geo='US')
        
        # Get interest over time
        interest_df = pytrends.interest_over_time()
        
        if interest_df.empty:
            return None
        
        # Calculate trend direction for each keyword
        results = {}
        for keyword in keywords[:5]:
            if keyword in interest_df.columns:
                series = interest_df[keyword]
                recent = series.tail(4).mean()  # Last 4 weeks
                earlier = series.head(4).mean()  # First 4 weeks
                
                if earlier > 0:
                    change_pct = ((recent - earlier) / earlier) * 100
                    if change_pct > 10:
                        direction = "growing"
                    elif change_pct < -10:
                        direction = "declining"
                    else:
                        direction = "stable"
                else:
                    direction = "stable"
                    change_pct = 0
                
                results[keyword] = {
                    "current_index": int(series.iloc[-1]),
                    "trend_direction": direction,
                    "change_pct": round(change_pct, 1),
                }
        
        # Get related queries
        try:
            related = pytrends.related_queries()
            for keyword in keywords[:5]:
                if keyword in related and related[keyword].get("rising") is not None:
                    rising = related[keyword]["rising"]
                    if rising is not None and not rising.empty:
                        results[keyword]["rising_queries"] = rising["query"].head(5).tolist()
        except:
            pass  # Related queries may fail sometimes
        
        return results
        
    except ImportError:
        print("pytrends not installed. Run: pip install pytrends")
        return None
    except Exception as e:
        print(f"Error fetching Google Trends: {e}")
        return None
