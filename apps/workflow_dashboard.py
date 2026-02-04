"""
Workflow Dashboard

The default landing page for Workflow ShelfGuard MVP.

Tabs:
1. Memo - Weekly War-Room Memo (Top 10 Episodes)
2. Action Queue - Ranked actions with owner/status/due dates
3. Alerts - High-signal alerts only
4. Scoreboard - Monthly metrics + $ Protected range
5. Diagnostics - Collapsed/secondary technical details
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# Import workflow modules
from src.models.episode import Episode, EpisodeStatus
from src.workflow.reason_codes import ReasonCode
from src.workflow.episode_builder import build_episodes
from src.workflow.action_templates import get_action_template, ActionTemplate, Urgency
from src.workflow.memo_renderer import render_weekly_memo, WeeklyMemo, MemoItem
from src.workflow.alerts import generate_alerts, Alert, get_alert_counts
from src.workflow.scoreboard import calculate_scoreboard, Scoreboard, render_scoreboard_summary
from src.workflow.config import get_category_config, get_reason_code_weight

# Import trigger detection (existing)
try:
    from src.trigger_detection import detect_trigger_events
    DETECTORS_AVAILABLE = True
except ImportError:
    detect_trigger_events = None
    DETECTORS_AVAILABLE = False


def render_workflow_dashboard():
    """
    Main entry point for the Workflow Dashboard.

    Renders tabs: Memo | Action Queue | Alerts | Scoreboard | Diagnostics
    """
    st.markdown("## 📋 Workflow Dashboard")

    # Check if project is loaded
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        _render_no_project_state()
        return

    # Get project data
    df_weekly = st.session_state.get("df_weekly", pd.DataFrame())
    df_competitors = st.session_state.get("df_competitors", pd.DataFrame())
    portfolio_asins = st.session_state.get("portfolio_asins", [])

    if df_weekly.empty:
        _render_no_data_state()
        return

    # Build episodes from trigger events
    episodes, alerts = _build_workflow_data(df_weekly, df_competitors, portfolio_asins)

    # Store in session state for persistence
    st.session_state["workflow_episodes"] = episodes
    st.session_state["workflow_alerts"] = alerts

    # Render tabs
    tab_memo, tab_actions, tab_alerts, tab_scoreboard, tab_diagnostics = st.tabs([
        "📝 Memo",
        "📋 Action Queue",
        "🚨 Alerts",
        "📊 Scoreboard",
        "🔧 Diagnostics"
    ])

    with tab_memo:
        _render_memo_tab(episodes)

    with tab_actions:
        _render_action_queue_tab(episodes)

    with tab_alerts:
        _render_alerts_tab(alerts)

    with tab_scoreboard:
        _render_scoreboard_tab(episodes, alerts)

    with tab_diagnostics:
        _render_diagnostics_tab(episodes, df_weekly, df_competitors)


def _render_no_project_state():
    """Render when no project is loaded."""
    st.markdown("""
    <div style="text-align: center; padding: 60px 40px; background: #f8f9fa; border-radius: 12px; border: 2px dashed #dee2e6;">
        <div style="font-size: 48px; margin-bottom: 20px;">📋</div>
        <div style="font-size: 24px; font-weight: 600; color: #495057; margin-bottom: 12px;">
            No Project Loaded
        </div>
        <div style="font-size: 14px; color: #6c757d;">
            Go to Market Discovery to create or load a project.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_no_data_state():
    """Render when project has no data."""
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: #fff3cd; border-radius: 8px;">
        <div style="font-size: 32px; margin-bottom: 12px;">⚠️</div>
        <div style="font-size: 18px; font-weight: 600; color: #856404;">
            No Data Available
        </div>
        <div style="font-size: 14px; color: #856404;">
            Run a data refresh to populate the dashboard.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _build_workflow_data(
    df_weekly: pd.DataFrame,
    df_competitors: pd.DataFrame,
    portfolio_asins: List[str]
) -> tuple[List[Episode], List[Alert]]:
    """
    Build Episodes and Alerts from raw data.

    This is the core pipeline: TriggerEvents → Episodes → Alerts
    """
    all_episodes = []
    all_alerts = []

    if not DETECTORS_AVAILABLE:
        return all_episodes, all_alerts

    # Get category config for weights
    module_id = st.session_state.get("category_module_id", "skincare_serum_moisturizer")
    config = get_category_config(module_id)
    category_weights = config.get("reason_code_weights", {})

    # Build weekly revenue map from df_weekly
    weekly_revenue_map = {}
    if "weekly_revenue" in df_weekly.columns and "asin" in df_weekly.columns:
        # Get most recent week's revenue per ASIN
        recent = df_weekly.sort_values("week_start", ascending=False).groupby("asin").first()
        for asin, row in recent.iterrows():
            weekly_revenue_map[asin] = row.get("weekly_revenue", 0) or 0

    # Run detectors for each portfolio ASIN
    for asin in portfolio_asins:
        # Filter historical data for this ASIN
        asin_history = df_weekly[df_weekly["asin"] == asin] if "asin" in df_weekly.columns else df_weekly

        if asin_history.empty:
            continue

        try:
            # Detect trigger events
            trigger_events = detect_trigger_events(
                asin=asin,
                df_historical=asin_history,
                df_competitors=df_competitors,
                lookback_days=30
            )

            if not trigger_events:
                continue

            # Build episodes from trigger events
            episodes = build_episodes(
                trigger_events=trigger_events,
                portfolio_asins=portfolio_asins,
                weekly_revenue_map=weekly_revenue_map,
                category_weights=category_weights,
            )

            all_episodes.extend(episodes)

        except Exception as e:
            # Log error but continue
            st.session_state.setdefault("workflow_errors", []).append(f"ASIN {asin}: {str(e)}")

    # Generate alerts from episodes
    all_alerts = generate_alerts(all_episodes, min_severity=0.8)

    return all_episodes, all_alerts


# =============================================================================
# TAB: MEMO
# =============================================================================

def _render_memo_tab(episodes: List[Episode]):
    """Render the Weekly War-Room Memo tab."""
    st.markdown("### 📝 Weekly War-Room Memo")

    if not episodes:
        st.info("No episodes detected this week. Check back after more data is collected.")
        return

    # Render memo
    memo = render_weekly_memo(episodes, min_severity=0.5, max_episodes=10)

    # Header stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Episodes", memo.episodes_filtered)
    with col2:
        st.metric("Threats", memo.threats_count)
    with col3:
        st.metric("Opportunities", memo.opportunities_count)
    with col4:
        st.metric("Est. Impact", memo.total_impact_range)

    st.markdown("---")

    # Render each memo item
    for item in memo.items:
        _render_memo_item(item)
        st.markdown("---")


def _render_memo_item(item: MemoItem):
    """Render a single memo item."""
    # Header with rank and headline
    col1, col2 = st.columns([1, 9])
    with col1:
        st.markdown(f"### #{item.rank}")
    with col2:
        st.markdown(f"### {item.headline}")

    # Summary
    st.markdown(item.summary)

    # Evidence
    st.caption(f"📊 Evidence: {item.evidence_summary}")

    # Impact and urgency
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Impact:** {item.impact_range}")
    with col2:
        st.markdown(f"**Urgency:** {item.urgency_emoji}")
    with col3:
        st.markdown(f"**Score:** {item.episode.composite_score:.2f}")

    # What to verify
    with st.expander("🔍 What to Verify"):
        for step in item.what_to_verify:
            st.markdown(f"- {step}")

    # Recommended action
    st.markdown(f"**Recommended Action:** {item.recommended_action}")


# =============================================================================
# TAB: ACTION QUEUE
# =============================================================================

def _render_action_queue_tab(episodes: List[Episode]):
    """Render the Action Queue tab."""
    st.markdown("### 📋 Action Queue")

    if not episodes:
        st.info("No actions pending.")
        return

    # Filter to actionable episodes (NEW or IN_PROGRESS)
    actionable = [e for e in episodes if e.status in [EpisodeStatus.NEW, EpisodeStatus.IN_PROGRESS]]

    if not actionable:
        st.success("All actions have been resolved or dismissed.")
        return

    # Sort by composite score
    actionable.sort(key=lambda e: -e.composite_score)

    # Render as table
    data = []
    for e in actionable:
        template = get_action_template(e.action_template_id)
        data.append({
            "Priority": f"#{actionable.index(e) + 1}",
            "Episode": e.episode_type.replace("_", " ").title(),
            "ASIN": e.primary_asins[0] if e.primary_asins else "-",
            "Action": template.title if template else "Review",
            "Owner": template.owner_role.value if template else "-",
            "Urgency": template.get_urgency_emoji() if template else "❓",
            "Due": (datetime.now() + timedelta(days=template.default_due_offset_days)).strftime("%b %d") if template else "-",
            "Status": e.get_status_emoji(),
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Action buttons
    st.markdown("---")
    st.markdown("**Quick Actions:**")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Mark Top Item Resolved"):
            if actionable:
                actionable[0].status = EpisodeStatus.RESOLVED
                actionable[0].resolved_at = datetime.now()
                st.success("Marked as resolved!")
                st.rerun()

    with col2:
        if st.button("❌ Dismiss Top Item"):
            if actionable:
                actionable[0].status = EpisodeStatus.DISMISSED
                st.info("Dismissed.")
                st.rerun()

    with col3:
        if st.button("📤 Export Queue"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"action_queue_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


# =============================================================================
# TAB: ALERTS
# =============================================================================

def _render_alerts_tab(alerts: List[Alert]):
    """Render the Alerts tab."""
    st.markdown("### 🚨 Alerts")

    if not alerts:
        st.success("No high-severity alerts. All clear!")
        return

    # Alert counts
    counts = get_alert_counts(alerts)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Alerts", counts["total"])
    with col2:
        st.metric("Critical", counts["critical"])
    with col3:
        st.metric("High", counts["high"])

    st.markdown("---")

    # Render each alert
    for alert in alerts:
        _render_alert_card(alert)


def _render_alert_card(alert: Alert):
    """Render a single alert card."""
    severity_color = "#dc3545" if alert.severity == "critical" else "#fd7e14"

    st.markdown(f"""
    <div style="border-left: 4px solid {severity_color}; padding: 12px 16px; margin-bottom: 12px; background: #f8f9fa; border-radius: 4px;">
        <div style="font-size: 16px; font-weight: 600;">
            {alert.get_severity_emoji()} {alert.title}
        </div>
        <div style="font-size: 14px; color: #495057; margin-top: 8px;">
            {alert.message}
        </div>
        <div style="font-size: 12px; color: #6c757d; margin-top: 8px;">
            ASINs: {', '.join(alert.primary_asins[:3])} |
            Created: {alert.created_at.strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        if not alert.is_acknowledged():
            if st.button("Acknowledge", key=f"ack_{alert.alert_id}"):
                alert.acknowledge()
                st.rerun()
        else:
            st.caption(f"✓ Acknowledged by {alert.acknowledged_by}")


# =============================================================================
# TAB: SCOREBOARD
# =============================================================================

def _render_scoreboard_tab(episodes: List[Episode], alerts: List[Alert]):
    """Render the Scoreboard tab."""
    st.markdown("### 📊 Monthly Scoreboard")

    # Calculate scoreboard
    scoreboard = calculate_scoreboard(episodes, alerts)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Episodes Detected", scoreboard.episodes_detected)
    with col2:
        st.metric("Resolved", scoreboard.episodes_resolved)
    with col3:
        st.metric("Alerts", scoreboard.high_severity_alerts)
    with col4:
        resolution_rate = scoreboard.get_resolution_rate() * 100
        st.metric("Resolution Rate", f"{resolution_rate:.0f}%")

    st.markdown("---")

    # $ Protected (with prominent assumptions)
    st.markdown("### 💰 $ Protected (Proxy Estimate)")
    st.markdown(f"## {scoreboard.get_protected_range_str()}")

    with st.expander("⚠️ Important: See Assumptions", expanded=True):
        for assumption in scoreboard.assumptions:
            st.markdown(f"- {assumption}")

    st.markdown("---")

    # Episodes by reason code
    st.markdown("### Episodes by Reason Code")
    if scoreboard.episodes_by_reason_code:
        df_reasons = pd.DataFrame([
            {"Reason Code": k.replace("_", " ").title(), "Count": v}
            for k, v in scoreboard.episodes_by_reason_code.items()
        ])
        st.bar_chart(df_reasons.set_index("Reason Code"))
    else:
        st.info("No episodes to display.")

    # Hours saved
    st.markdown("---")
    st.markdown(f"### ⏱️ Estimated Hours Saved: {scoreboard.hours_saved_estimate:.1f}")
    st.caption(" | ".join(scoreboard.hours_saved_assumptions))


# =============================================================================
# TAB: DIAGNOSTICS
# =============================================================================

def _render_diagnostics_tab(episodes: List[Episode], df_weekly: pd.DataFrame, df_competitors: pd.DataFrame):
    """Render the Diagnostics tab (collapsed by default)."""
    st.markdown("### 🔧 Diagnostics")
    st.caption("Technical details for debugging. Hidden by default in production.")

    # Data summary
    with st.expander("📊 Data Summary"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Weekly Rows", len(df_weekly))
        with col2:
            st.metric("Competitor Rows", len(df_competitors))
        with col3:
            st.metric("Episodes Built", len(episodes))

    # Episode details
    with st.expander("📝 Episode Details"):
        if episodes:
            episode_data = [e.to_dict() for e in episodes[:20]]
            st.json(episode_data)
        else:
            st.info("No episodes.")

    # Errors
    errors = st.session_state.get("workflow_errors", [])
    if errors:
        with st.expander("❌ Errors"):
            for error in errors[-10:]:
                st.error(error)

    # Config
    with st.expander("⚙️ Active Configuration"):
        module_id = st.session_state.get("category_module_id", "skincare_serum_moisturizer")
        config = get_category_config(module_id)
        st.json({
            "module_id": config.get("module_id"),
            "display_name": config.get("display_name"),
            "thresholds": config.get("thresholds"),
            "reason_code_weights": config.get("reason_code_weights"),
        })
