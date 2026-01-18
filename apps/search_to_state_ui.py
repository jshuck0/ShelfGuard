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
    Main Discovery UI component with Two-Phase Architecture.

    Phase 1: Lightweight seed discovery (25-50 results)
    Phase 2: Dynamic category mapping (fetch until 90% revenue captured)

    Renders:
    1. Keyword search for seed products
    2. Seed product selection
    3. Dynamic category market mapping
    4. Market snapshot visualization
    5. Pin to State button
    """
    st.markdown("### 🔍 Two-Phase Market Discovery")
    st.markdown("**Phase 1**: Find seed products → **Phase 2**: Map full competitive market")

    # Input mode selector
    input_mode = st.radio(
        "Search Method",
        ["🔎 Two-Phase Discovery", "📋 Manual ASINs"],
        horizontal=True
    )

    if input_mode == "📋 Manual ASINs":
        # Manual ASIN input (legacy path)
        asin_text = st.text_area(
            "Enter ASINs (one per line or comma-separated)",
            placeholder="B07GMLSQG5\nB0928F3QZ7\nB07PQLHFQ1",
            height=150
        )

        if not asin_text:
            st.info("💡 Paste ASINs above to analyze specific products")
            return

        # Parse ASINs
        raw_asins = asin_text.replace(",", "\n").split("\n")
        asins_to_analyze = [a.strip().upper() for a in raw_asins if a.strip()]
        st.caption(f"✅ {len(asins_to_analyze)} ASINs ready to analyze")

        if not st.button("🚀 Analyze Products", type="primary"):
            return

        # Fetch using legacy path (no category context)
        with st.spinner(f"📊 Fetching data for {len(asins_to_analyze)} products..."):
            try:
                from src.discovery import fetch_asins_from_keepa, prune_to_90_percent

                market_snapshot = fetch_asins_from_keepa(asins_to_analyze)

                if market_snapshot.empty:
                    st.error("❌ No data returned from Keepa.")
                    return

                market_snapshot, stats = prune_to_90_percent(market_snapshot)
                stats["query"] = f"{len(asins_to_analyze)} Products"
                stats["total_asins"] = len(asins_to_analyze)

            except Exception as e:
                st.error(f"❌ Data fetch failed: {str(e)}")
                st.exception(e)
                return

    else:  # Two-Phase Discovery
        # ========== CATEGORY SELECTION (OPTIONAL) ==========
        st.markdown("#### 🎯 Step 1: Define Your Market")

        search_mode = st.radio(
            "How would you like to search?",
            ["🔍 Keyword Search (I'm exploring)", "📂 Category-First (I know my market)"],
            horizontal=True,
            help="Category-first gives cleaner results for known markets"
        )

        category_filter = None

        if search_mode == "📂 Category-First (I know my market)":
            # Category selector for power users
            st.markdown("**Select Amazon Category:**")

            # Common categories for CPG/ecommerce users
            category_options = {
                "Grocery & Gourmet Food": 16310101,
                "Health & Household": 3760901,
                "Beauty & Personal Care": 3760911,
                "Pet Supplies": 2619533011,
                "Home & Kitchen": 1055398,
                "Sports & Outdoors": 3375251,
                "Baby Products": 165796011,
                "Toys & Games": 165793011,
                "Electronics": 172282,
                "Clothing, Shoes & Jewelry": 7141123011,
                "Office Products": 1064954,
                "Industrial & Scientific": 16310091,
            }

            selected_category = st.selectbox(
                "Amazon Root Category",
                options=list(category_options.keys()),
                help="Select the primary market category for your analysis"
            )

            category_filter = category_options[selected_category]
            st.caption(f"📂 Category ID: {category_filter}")

        # ========== PHASE 1: SEED DISCOVERY ==========
        st.markdown("#### 🌱 Phase 1: Find Seed Product")

        col1, col2 = st.columns([3, 1])

        with col1:
            if category_filter:
                search_keyword = st.text_input(
                    f"Search within {selected_category}",
                    placeholder="e.g., Starbucks, organic, premium brand",
                    help="Search for products within the selected category"
                )
            else:
                search_keyword = st.text_input(
                    "Search Query",
                    placeholder="e.g., Windex, Starbucks, Kraft, almond milk",
                    help="Search for products to find a seed product that defines your market"
                )

        with col2:
            seed_limit = st.selectbox(
                "Seed Results",
                [10, 25, 50],
                index=1,
                help="Number of seed candidates to show"
            )

        if not search_keyword:
            st.info("💡 Enter a search term to find seed products (e.g., 'Windex', 'Starbucks', 'Dunkin')")
            return

        # Phase 1 Search Button
        search_key = f"{search_keyword}_{category_filter}"  # Unique key per category
        if "seed_products_df" not in st.session_state or st.session_state.get("last_search") != search_key:
            if st.button("🔍 Find Seed Products", type="primary"):
                with st.spinner(f"🌱 Searching for '{search_keyword}' seed products..."):
                    try:
                        from src.two_phase_discovery import phase1_seed_discovery

                        seed_df = phase1_seed_discovery(
                            keyword=search_keyword,
                            limit=seed_limit,
                            domain="US",
                            category_filter=category_filter  # Pass category filter
                        )

                        if seed_df.empty:
                            st.error(f"❌ No seed products found for '{search_keyword}'")
                            return

                        st.session_state["seed_products_df"] = seed_df
                        st.session_state["last_search"] = search_key
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Phase 1 failed: {str(e)}")
                        st.exception(e)
                        return

        # Display seed products if available
        if "seed_products_df" in st.session_state:
            seed_df = st.session_state["seed_products_df"]

            st.success(f"✅ Found {len(seed_df)} seed candidates for '{search_keyword}'")

            # Check for category ambiguity (keyword-only mode)
            if not category_filter and not seed_df.empty and "category_id" in seed_df.columns:
                unique_categories = seed_df["category_id"].nunique()
                if unique_categories > 2:
                    # Show disambiguation warning
                    top_categories = seed_df.groupby("category_path").size().sort_values(ascending=False).head(4)

                    st.warning(
                        f"⚠️ **Multiple Markets Detected**: Found products in {unique_categories} different categories.\n\n"
                        f"For better results, consider using **Category-First** mode above."
                    )

                    with st.expander("📂 See Category Breakdown"):
                        for cat_path, count in top_categories.items():
                            st.caption(f"• {cat_path}: {count} products")

            # Show category breadcrumb from first result
            if not seed_df.empty and "category_path" in seed_df.columns:
                st.caption(f"📂 Category: **{seed_df.iloc[0]['category_path']}**")

            # Seed selection table
            st.markdown("**Select a seed product to define the market:**")

            # Format for display
            display_df = seed_df.head(20)[["title", "brand", "price", "bsr", "category_path"]].copy()
            display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")
            display_df["bsr"] = display_df["bsr"].apply(lambda x: f"{int(x):,}" if x > 0 else "N/A")

            # Radio button selection
            if len(seed_df) > 0:
                selected_idx = st.radio(
                    "Choose seed product:",
                    range(min(20, len(seed_df))),
                    format_func=lambda i: f"{seed_df.iloc[i]['title'][:80]}... | {seed_df.iloc[i]['brand']} | BSR: {int(seed_df.iloc[i]['bsr']):,}",
                    label_visibility="collapsed"
                )

                seed_product = seed_df.iloc[selected_idx]

                # ========== PHASE 2: CATEGORY MAPPING ==========
                st.markdown("---")
                st.markdown("#### 🗺️ Phase 2: Map Competitive Market")

                st.info(
                    f"**Seed**: {seed_product['title'][:100]}\n\n"
                    f"**Category**: {seed_product['category_path']}\n\n"
                    f"ShelfGuard will now fetch products from this category until 90% of revenue is captured."
                )

                col1, col2 = st.columns([2, 1])

                with col1:
                    if st.button("🚀 Map Full Market", type="primary"):
                        st.session_state["trigger_phase2"] = True
                        st.rerun()

                with col2:
                    max_products = st.number_input(
                        "Max Products",
                        min_value=100,
                        max_value=1000,
                        value=500,
                        step=100,
                        help="Safety limit for category fetch"
                    )

                # Execute Phase 2 if triggered
                if st.session_state.get("trigger_phase2"):
                    with st.spinner("🗺️ Mapping competitive market (dynamic 90% fetch)..."):
                        try:
                            from src.two_phase_discovery import phase2_category_market_mapping

                            market_snapshot, market_stats = phase2_category_market_mapping(
                                category_id=int(seed_product["category_id"]),
                                seed_product_title=seed_product["title"],
                                target_revenue_pct=90.0,
                                max_products=max_products,
                                batch_size=100,
                                domain="US"
                            )

                            if market_snapshot.empty:
                                st.error("❌ No products found in category")
                                return

                            # Prepare stats for display
                            stats = {
                                "query": f'"{search_keyword}" ({seed_product["category_path"]})',
                                "category": seed_product["category_path"],
                                "total_asins": market_stats["total_products"],
                                "pruned_asins": market_stats["validated_products"],
                                "pruned_pct": (market_stats["validated_products"] / market_stats["total_products"] * 100) if market_stats["total_products"] > 0 else 0,
                                "revenue_captured_pct": 90.0,  # By design
                                "total_revenue_proxy": market_stats["validated_revenue"]
                            }

                            # Clear trigger
                            st.session_state["trigger_phase2"] = False

                        except Exception as e:
                            st.error(f"❌ Phase 2 failed: {str(e)}")
                            st.exception(e)
                            st.session_state["trigger_phase2"] = False
                            return
                else:
                    return  # Wait for Phase 2 trigger
            else:
                return
        else:
            return  # Wait for Phase 1 search

    # ========== RESULTS VISUALIZATION ==========
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
