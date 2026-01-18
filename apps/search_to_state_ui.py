"""
ShelfGuard Search-to-State UI Components
==========================================
Drop-in components for integrating Search-to-State into shelfguard_app.py

Usage:
    from apps.search_to_state_ui import render_discovery_ui

    # In sidebar or main area:
    render_discovery_ui()
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Import core modules
from src.discovery import execute_market_discovery
from src.persistence import (
    pin_to_state,
    get_mission_profile_config,
    load_user_projects,
    load_project_asins
)
from src.backfill import execute_backfill
from src.recommendations import generate_resolution_cards, render_resolution_card


def render_discovery_ui() -> None:
    """
    Main Discovery UI component.

    Renders:
    1. Keyword/brand search OR manual ASIN input
    2. Market snapshot visualization
    3. Pin to State button
    4. Mission profile selector
    """
    st.markdown("### 🔍 Market Discovery")
    st.markdown("Search for products by keyword, brand, or category to discover market opportunities.")

    # Input mode selector
    input_mode = st.radio(
        "Search Method",
        ["🔎 Keyword Search", "📋 Manual ASINs"],
        horizontal=True
    )

    asins_to_analyze = []
    search_keyword = None

    if input_mode == "🔎 Keyword Search":
        # Keyword search input
        col1, col2 = st.columns([3, 1])

        with col1:
            search_keyword = st.text_input(
                "Search Query",
                placeholder="e.g., Starbucks, Dunkin, almond milk, organic coffee",
                help="Search for products by brand name, keyword, or category"
            )

        with col2:
            max_results = st.selectbox(
                "Max Results",
                [100, 250, 500],
                index=2,
                help="Maximum number of products to fetch"
            )

        if not search_keyword:
            st.info("💡 Enter a search term above (e.g., 'Starbucks', 'Kraft', 'Dunkin', 'almond')")
            return

        # Search button
        if not st.button("🔍 Search Products", type="primary"):
            return

        # Execute search
        with st.spinner(f"🔎 Searching Keepa for '{search_keyword}'..."):
            try:
                from src.discovery import search_products_by_keyword

                asins_to_analyze = search_products_by_keyword(
                    keyword=search_keyword,
                    limit=max_results,
                    domain="US"
                )

                if not asins_to_analyze:
                    st.error(f"❌ No products found for '{search_keyword}'. Try a different keyword.")
                    return

            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                st.exception(e)
                return

    else:  # Manual ASIN input
        asin_text = st.text_area(
            "Enter ASINs (one per line or comma-separated)",
            placeholder="B07GMLSQG5\nB0928F3QZ7\nB07PQLHFQ1",
            height=150
        )

        if asin_text:
            # Parse ASINs - support both newline and comma separation
            raw_asins = asin_text.replace(",", "\n").split("\n")
            asins_to_analyze = [a.strip().upper() for a in raw_asins if a.strip()]

            st.caption(f"✅ {len(asins_to_analyze)} ASINs ready to analyze")
        else:
            st.info("💡 Paste ASINs above to analyze specific products")
            return

        # Analyze button
        if not st.button("🚀 Analyze Products", type="primary"):
            return

    # Fetch product data from Keepa
    with st.spinner(f"📊 Fetching data for {len(asins_to_analyze)} products from Keepa..."):
        try:
            from src.discovery import fetch_asins_from_keepa, prune_to_90_percent

            market_snapshot = fetch_asins_from_keepa(asins_to_analyze)

            if market_snapshot.empty:
                st.error("❌ No data returned from Keepa. Check your API key or ASINs.")
                return

            # Prune to 90%
            market_snapshot, stats = prune_to_90_percent(market_snapshot)

            # Set display name
            if search_keyword:
                stats["query"] = f'"{search_keyword}"'
            else:
                stats["query"] = f"{len(asins_to_analyze)} Products"

            stats["total_asins"] = len(asins_to_analyze)

        except Exception as e:
            st.error(f"❌ Data fetch failed: {str(e)}")
            st.exception(e)
            return

    # Display stats
    st.markdown("---")
    st.markdown(f"### 📊 Market Snapshot: **{stats['query']}**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total ASINs Scanned",
            f"{stats['total_asins']:,}",
            delta=None
        )

    with col2:
        st.metric(
            "Pruned Set (90% Rule)",
            f"{stats['pruned_asins']:,}",
            delta=f"{stats['pruned_pct']:.0f}% of total"
        )

    with col3:
        st.metric(
            "Revenue Captured",
            f"{stats['revenue_captured_pct']:.1f}%",
            delta="Target: 90%"
        )

    with col4:
        st.metric(
            "Total Revenue Proxy",
            f"${stats['total_revenue_proxy']:,.0f}",
            delta=None
        )

    # Visualizations
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Market Share", "💰 Price vs BSR", "📋 Top 20 ASINs"])

    with tab1:
        # Donut chart showing revenue distribution
        fig = px.pie(
            market_snapshot.head(20),  # Top 20 for readability
            values="revenue_proxy",
            names="title",
            title="Revenue Distribution (Top 20 ASINs)",
            hole=0.4
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Scatter plot: Price vs BSR
        fig = px.scatter(
            market_snapshot,
            x="bsr",
            y="price",
            size="monthly_units",
            color="revenue_proxy",
            hover_data=["title", "asin"],
            title="Price vs Sales Rank (BSR)",
            labels={
                "bsr": "Best Sellers Rank",
                "price": "Current Price ($)",
                "monthly_units": "Monthly Units Sold"
            },
            color_continuous_scale="Viridis"
        )
        fig.update_xaxes(type="log")  # Log scale for BSR
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Data table
        top_20 = market_snapshot.head(20)[[
            "asin", "title", "price", "monthly_units", "revenue_proxy", "bsr"
        ]].copy()

        top_20["revenue_proxy"] = top_20["revenue_proxy"].apply(lambda x: f"${x:,.0f}")
        top_20["price"] = top_20["price"].apply(lambda x: f"${x:.2f}")
        top_20["bsr"] = top_20["bsr"].apply(lambda x: f"{x:,.0f}")

        st.dataframe(
            top_20,
            use_container_width=True,
            hide_index=True,
            column_config={
                "asin": "ASIN",
                "title": "Product Title",
                "price": "Price",
                "monthly_units": "Monthly Units",
                "revenue_proxy": "Est. Monthly Revenue",
                "bsr": "Sales Rank"
            }
        )

    # Pin to State section
    st.markdown("---")
    render_pin_to_state_ui(market_snapshot, stats)


def render_pin_to_state_ui(market_snapshot: pd.DataFrame, stats: dict) -> None:
    """
    UI for creating a new project from discovered ASINs.

    Args:
        market_snapshot: Pruned DataFrame from discovery
        stats: Discovery statistics
    """
    st.markdown("### 📌 Pin to State (Create Project)")

    col1, col2 = st.columns([2, 1])

    with col1:
        project_name = st.text_input(
            "Project Name",
            value=f"{stats.get('query', 'Market')} Monitoring",
            placeholder="e.g., Starbucks K-Cups Q1 2026"
        )

    with col2:
        # Mission profile selector
        mission_options = {
            "bodyguard": "🛡️ Bodyguard (Defensive)",
            "scout": "🔍 Scout (Offensive)",
            "surgeon": "🔬 Surgeon (Efficiency)"
        }

        mission_type = st.selectbox(
            "Mission Profile",
            options=list(mission_options.keys()),
            format_func=lambda x: mission_options[x]
        )

    # Show mission profile details
    config = get_mission_profile_config(mission_type)
    with st.expander(f"ℹ️ About {config['name']}"):
        st.markdown(f"**Focus**: {config['focus']}")
        st.markdown("**Top Priorities:**")
        top_3 = sorted(config["priorities"].items(), key=lambda x: x[1], reverse=True)[:3]
        for priority, weight in top_3:
            st.markdown(f"- {priority.replace('_', ' ').title()} (weight: {weight})")

    # Pin button
    if st.button("🚀 Create Project & Backfill History", type="primary"):
        with st.spinner("Creating project and fetching 90-day history..."):
            try:
                # Create project
                project_id = pin_to_state(
                    asins=market_snapshot["asin"].tolist(),
                    project_name=project_name,
                    mission_type=mission_type,
                    user_id=None,  # TODO: Integrate with auth.uid()
                    metadata={
                        **stats,
                        "search_date": pd.Timestamp.now().isoformat()
                    }
                )

                # Trigger backfill (async)
                execute_backfill(
                    project_id=project_id,
                    asins=market_snapshot["asin"].tolist(),
                    run_async=True
                )

                st.success(
                    f"✅ Project '{project_name}' created!\n\n"
                    f"📊 Tracking {len(market_snapshot)} ASINs\n"
                    f"🔄 Historical backfill in progress (90 days of Price & BSR data)\n"
                    f"🎯 Mission: {config['name']}"
                )

                # Store project_id in session state for navigation
                st.session_state["active_project_id"] = project_id
                st.session_state["show_project_dashboard"] = True

            except Exception as e:
                st.error(f"❌ Failed to create project: {str(e)}")


def render_project_dashboard(project_id: str) -> None:
    """
    Project-specific dashboard showing alerts and metrics.

    Args:
        project_id: UUID of the project to display
    """
    from supabase import create_client

    # Fetch project details
    supabase = create_client(st.secrets["url"], st.secrets["key"])

    try:
        # Get project metadata
        project_result = supabase.table("projects").select("*").eq(
            "id", project_id
        ).execute()

        if not project_result.data:
            st.error("Project not found")
            return

        project = project_result.data[0]

        # Header
        st.markdown(f"## {project['project_name']}")

        config = get_mission_profile_config(project["mission_type"])
        st.markdown(f"**Mission Profile**: {config['name']} | **ASINs**: {project['asin_count']}")

        # Fetch historical metrics
        metrics_result = supabase.table("historical_metrics").select("*").eq(
            "project_id", project_id
        ).execute()

        if not metrics_result.data:
            st.info("🔄 Historical backfill in progress. Charts will appear shortly.")
            return

        df_metrics = pd.DataFrame(metrics_result.data)

        # Generate resolution cards
        st.markdown("---")
        st.markdown("### 🎯 Resolution Cards")

        # TODO: Fetch current week data from engine.run_weekly_analysis()
        # For now, use empty DataFrame
        df_current = pd.DataFrame()

        alerts = generate_resolution_cards(
            df_metrics=df_metrics,
            df_current=df_current,
            mission_type=project["mission_type"]
        )

        if alerts:
            for alert in alerts[:10]:  # Top 10
                render_resolution_card(alert)
        else:
            st.success("✅ No critical alerts at this time. Portfolio is healthy.")

        # Historical charts
        st.markdown("---")
        st.markdown("### 📈 Historical Trends")

        # BSR trend
        if "sales_rank" in df_metrics.columns:
            fig = px.line(
                df_metrics.groupby("datetime")["sales_rank"].mean().reset_index(),
                x="datetime",
                y="sales_rank",
                title="Average Sales Rank (Lower = Better)",
                labels={"datetime": "Date", "sales_rank": "BSR"}
            )
            fig.update_yaxes(autorange="reversed")  # Lower BSR = better
            st.plotly_chart(fig, use_container_width=True)

        # Price trend
        if "buy_box_price" in df_metrics.columns:
            fig = px.line(
                df_metrics.groupby("datetime")["buy_box_price"].mean().reset_index(),
                x="datetime",
                y="buy_box_price",
                title="Average Buy Box Price",
                labels={"datetime": "Date", "buy_box_price": "Price ($)"}
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load project dashboard: {str(e)}")


def render_project_selector() -> Optional[str]:
    """
    Sidebar component for selecting existing projects.

    Returns:
        Selected project_id or None
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Your Projects")

    # Load user projects
    projects_df = load_user_projects(user_id=None)  # TODO: Get from auth

    if projects_df.empty:
        st.sidebar.info("No projects yet. Use Discovery to create one.")
        return None

    # Project dropdown
    project_options = {
        row["id"]: f"{row['project_name']} ({row['asin_count']} ASINs)"
        for _, row in projects_df.iterrows()
    }

    selected_id = st.sidebar.selectbox(
        "Select Project",
        options=list(project_options.keys()),
        format_func=lambda x: project_options[x],
        label_visibility="collapsed"
    )

    return selected_id


# Example integration with shelfguard_app.py:
"""
# In apps/shelfguard_app.py:

from apps.search_to_state_ui import (
    render_discovery_ui,
    render_project_dashboard,
    render_project_selector
)

# Add new tab to main navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Current Dashboard",
    "🔍 Market Discovery",  # NEW
    "📂 Projects",          # NEW
    "💬 AI Chat"
])

with tab2:
    render_discovery_ui()

with tab3:
    project_id = render_project_selector()
    if project_id:
        render_project_dashboard(project_id)
"""
