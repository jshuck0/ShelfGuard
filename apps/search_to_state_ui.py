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
from src.persistence import (
    pin_to_state,
    get_mission_profile_config,
    load_user_projects,
    load_project_asins
)
from src.backfill import execute_backfill
from src.recommendations import generate_resolution_cards, render_resolution_card

# Import cache function for instant return visits
try:
    from src.supabase_reader import cache_market_snapshot
    CACHE_ENABLED = True
except ImportError:
    cache_market_snapshot = None
    CACHE_ENABLED = False


def render_discovery_ui() -> None:
    """
    Main Discovery UI component with Two-Phase Architecture.

    Phase 1: Lightweight seed discovery (25-50 results)
    Phase 2: Dynamic category mapping (fetch until 80% revenue captured)

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
        # ========== SIMPLIFIED DISCOVERY ==========
        st.markdown("#### 🎯 Step 1: Define Your Market")

        search_mode = st.radio(
            "How would you like to search?",
            ["🔍 Keyword Search", "📂 Category + Keyword"],
            horizontal=True,
            help="Use 'Category + Keyword' for cleaner results when you know your market"
        )
        
        # Always use Family Harvester (best behavior) - no complex selector needed
        use_family_harvester = True

        category_filter = None

        if search_mode == "📂 Category + Keyword":
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
                    placeholder="e.g., RXBAR, Starbucks, Kraft, organic",
                    help="Search for products within the selected category",
                    key="search_keyword_with_category"
                )
            else:
                search_keyword = st.text_input(
                    "Search (Brand or Keyword)",
                    placeholder="e.g., RXBAR, Poppi, Starbucks, almond milk",
                    help="Search by brand name or product keyword",
                    key="search_keyword_general"
                )

        with col2:
            seed_limit = st.selectbox(
                "Results",
                [5, 10, 25, 50],
                index=1,
                help="Number of seed products to show"
            )

        if not search_keyword:
            st.info("💡 Enter a brand or keyword (e.g., 'RXBAR', 'Poppi', 'protein bars')")
            return

        # Phase 1 Search Button
        search_key = f"{search_keyword}_{category_filter}"
        if "seed_products_df" not in st.session_state or st.session_state.get("last_search") != search_key:
            if st.button("🔍 Find Products", type="primary"):
                # Clear previous Phase 2 data when starting a new search
                if "discovery_market_snapshot" in st.session_state:
                    del st.session_state["discovery_market_snapshot"]
                if "discovery_stats" in st.session_state:
                    del st.session_state["discovery_stats"]
                    
                with st.spinner(f"🔍 Searching for '{search_keyword}'..."):
                    try:
                        from src.two_phase_discovery import phase1_seed_discovery

                        seed_df = phase1_seed_discovery(
                            keyword=search_keyword,
                            limit=seed_limit,
                            domain="US",
                            category_filter=category_filter,
                            use_family_harvester=True  # Always use best discovery
                        )

                        if seed_df.empty:
                            st.error(f"❌ No products found for '{search_keyword}'")
                            return

                        st.session_state["seed_products_df"] = seed_df
                        st.session_state["last_search"] = search_key
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Discovery failed: {str(e)}")
                        st.exception(e)
                        return

        # Display seed products if available
        if "seed_products_df" in st.session_state:
            seed_df = st.session_state["seed_products_df"]
            
            # Check if we used family harvester (has family_size column)
            used_families = "family_size" in seed_df.columns and "parent_asin" in seed_df.columns
            
            if used_families:
                unique_families = seed_df["parent_asin"].nunique() if "parent_asin" in seed_df.columns else 1
                st.success(
                    f"✅ Found {len(seed_df)} products across {unique_families} product families "
                    f"for '{search_keyword}'"
                )
                
                # Show family breakdown
                with st.expander("🧬 Product Family Breakdown", expanded=False):
                    if "parent_asin" in seed_df.columns:
                        # Build agg dict dynamically based on available columns
                        agg_dict = {
                            "asin": "count",
                            "brand": "first",
                        }
                        # Add title column if available
                        if "family_title" in seed_df.columns:
                            agg_dict["family_title"] = "first"
                            title_col = "family_title"
                        elif "title" in seed_df.columns:
                            agg_dict["title"] = "first"
                            title_col = "title"
                        else:
                            title_col = None
                        
                        family_summary = seed_df.groupby("parent_asin").agg(agg_dict).reset_index()
                        
                        # Rename columns
                        if title_col:
                            family_summary.columns = ["Parent ASIN", "Variations", "Brand", "Title"]
                        else:
                            family_summary.columns = ["Parent ASIN", "Variations", "Brand"]
                            family_summary["Title"] = family_summary["Brand"]  # Fallback
                        
                        family_summary = family_summary.sort_values("Variations", ascending=False)
                        st.dataframe(family_summary.head(10), use_container_width=True)
            else:
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

                seed_brand = seed_product.get("brand", "")
                
                st.info(
                    f"**Seed**: {seed_product['title'][:100]}\n\n"
                    f"**Target Brand**: {seed_brand}\n\n"
                    f"**Category**: {seed_product['category_path']}\n\n"
                    f"ShelfGuard will fetch ALL **{seed_brand}** products first, then fill with competitors."
                )
                
                if st.button("🚀 Map Full Market", type="primary"):
                    st.session_state["trigger_phase2"] = True
                    st.session_state["target_brand"] = seed_brand  # Pass brand to Phase 2
                    st.rerun()

                # Execute Phase 2 if triggered
                if st.session_state.get("trigger_phase2"):
                    with st.spinner("🗺️ Mapping competitive market (fetching 100 ASINs)..."):
                        try:
                            from src.two_phase_discovery import phase2_category_market_mapping

                            # Extract category info for progressive filtering
                            # category_tree_ids enables walking up hierarchy until 20+ products found
                            leaf_category_id = seed_product.get("leaf_category_id")
                            category_path = seed_product.get("category_path")
                            category_tree_ids_list = seed_product.get("category_tree_ids")
                            # Convert to tuple for caching (lists aren't hashable)
                            category_tree_ids = tuple(category_tree_ids_list) if category_tree_ids_list else None
                            
                            # Get target brand from session state
                            target_brand = st.session_state.get("target_brand", seed_brand)
                            
                            market_snapshot, market_stats = phase2_category_market_mapping(
                                category_id=int(seed_product["category_id"]),
                                seed_product_title=seed_product["title"],
                                seed_asin=seed_product["asin"],
                                target_revenue_pct=80.0,
                                max_products=100,
                                batch_size=100,
                                domain="US",
                                leaf_category_id=int(leaf_category_id) if leaf_category_id else None,
                                category_path=category_path,
                                category_tree_ids=category_tree_ids,
                                target_brand=target_brand  # Fetch ALL brand products first
                            )

                            if market_snapshot.empty:
                                st.error("❌ No products found in category")
                                return

                            # Prepare stats for display
                            # Use effective_category_path from market_stats (set by progressive fallback)
                            effective_path = market_stats.get("effective_category_path", seed_product["category_path"])
                            stats = {
                                "query": f'"{search_keyword}" ({effective_path})',
                                "category": effective_path,
                                "total_asins": market_stats["total_products"],
                                "pruned_asins": market_stats["validated_products"],
                                "pruned_pct": (market_stats["validated_products"] / market_stats["total_products"] * 100) if market_stats["total_products"] > 0 else 0,
                                "revenue_captured_pct": 80.0,  # By design
                                "total_revenue_proxy": market_stats["validated_revenue"],
                                "effective_category_id": market_stats.get("effective_category_id"),
                                "df_weekly": market_stats.get("df_weekly"),  # Include weekly data from phase2
                                "time_period": market_stats.get("time_period", "90 days"),  # Time period represented
                            }

                            # Store in session state so they persist across reruns
                            st.session_state["discovery_market_snapshot"] = market_snapshot
                            st.session_state["discovery_stats"] = stats

                            # Store search parameters for cache restoration
                            st.session_state["last_phase2_params"] = {
                                "category_id": int(seed_product["category_id"]),
                                "seed_product_title": seed_product["title"],
                                "seed_product_asin": seed_product["asin"],
                                "leaf_category_id": int(leaf_category_id) if leaf_category_id else None,
                                "category_path": category_path,
                                "category_tree_ids": category_tree_ids,
                                "search_keyword": search_keyword,
                                "target_brand": target_brand  # Store brand for cache
                            }

                            # Clear trigger
                            st.session_state["trigger_phase2"] = False

                        except Exception as e:
                            st.error(f"❌ Phase 2 failed: {str(e)}")
                            st.exception(e)
                            st.session_state["trigger_phase2"] = False
                            return
                else:
                    # Check if we have data from a previous Phase 2 run
                    if "discovery_market_snapshot" not in st.session_state:
                        # Try to restore from cache using stored parameters
                        if "last_phase2_params" in st.session_state:
                            params = st.session_state["last_phase2_params"]
                            try:
                                from src.two_phase_discovery import phase2_category_market_mapping
                                # This will use cached data if available (instant)
                                market_snapshot, market_stats = phase2_category_market_mapping(
                                    category_id=params["category_id"],
                                    seed_product_title=params["seed_product_title"],
                                    seed_asin=params.get("seed_product_asin"),
                                    target_revenue_pct=80.0,
                                    max_products=100,
                                    batch_size=100,
                                    domain="US",
                                    leaf_category_id=params["leaf_category_id"],
                                    category_path=params["category_path"],
                                    category_tree_ids=params["category_tree_ids"],
                                    target_brand=params.get("target_brand")  # Restore brand for cache
                                )
                                if not market_snapshot.empty:
                                    # Restore data from cache
                                    # Use effective_category_path from market_stats (set by progressive fallback)
                                    effective_path = market_stats.get("effective_category_path", params["category_path"])
                                    stats = {
                                        "query": f'"{params["search_keyword"]}" ({effective_path})',
                                        "category": effective_path,
                                        "total_asins": market_stats["total_products"],
                                        "pruned_asins": market_stats["validated_products"],
                                        "pruned_pct": (market_stats["validated_products"] / market_stats["total_products"] * 100) if market_stats["total_products"] > 0 else 0,
                                        "revenue_captured_pct": 80.0,
                                        "total_revenue_proxy": market_stats["validated_revenue"],
                                        "effective_category_id": market_stats.get("effective_category_id"),
                                        "df_weekly": market_stats.get("df_weekly"),  # Include weekly data from phase2
                                        "time_period": market_stats.get("time_period", "90 days"),  # Time period represented
                                    }
                                    st.session_state["discovery_market_snapshot"] = market_snapshot
                                    st.session_state["discovery_stats"] = stats
                                    st.info("🔄 Restored previous market mapping from cache")
                            except Exception as e:
                                pass  # Cache miss or error - user needs to search again
                        
                        if "discovery_market_snapshot" not in st.session_state:
                            return  # Wait for Phase 2 trigger
            else:
                if "discovery_market_snapshot" not in st.session_state:
                    return
        else:
            if "discovery_market_snapshot" not in st.session_state:
                return  # Wait for Phase 1 search

    # ========== RESULTS VISUALIZATION ==========
    # Retrieve from session state (persists across reruns)
    if "discovery_market_snapshot" in st.session_state:
        market_snapshot = st.session_state["discovery_market_snapshot"]
        stats = st.session_state["discovery_stats"]
    else:
        # No data available - shouldn't reach here but safety check
        return
    
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
            delta="Target: 80%"
        )

    with col4:
        time_period = stats.get("time_period", "current")
        st.metric(
            "Monthly Revenue (90-day avg)" if "90" in time_period else "Monthly Revenue Est.",
            f"${stats['total_revenue_proxy']:,.0f}",
            delta=f"Based on {time_period}" if time_period != "current" else None
        )

    # Visualizations
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Market Share", "💰 Price vs BSR", "📋 Top 20 ASINs"])

    with tab1:
        # Donut chart showing revenue distribution
        # Clean data: remove rows with NaN revenue_proxy
        pie_data = market_snapshot.head(20).copy()
        pie_data = pie_data.dropna(subset=["revenue_proxy", "title"])
        pie_data = pie_data[pie_data["revenue_proxy"] > 0]  # Only show products with revenue
        
        if pie_data.empty:
            st.warning("⚠️ No valid revenue data for pie chart")
        else:
            fig = px.pie(
                pie_data,
                values="revenue_proxy",
                names="title",
                title="Revenue Distribution (Top 20 ASINs)",
                hole=0.4
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="discovery_market_share_chart")

    with tab2:
        # Scatter plot: Price vs BSR
        # Clean data: remove rows with NaN in required columns
        scatter_data = market_snapshot[[
            "bsr", "price", "monthly_units", "revenue_proxy", "title", "asin"
        ]].copy()
        
        # Fill NaN values for size (monthly_units) with 1 to avoid plotly errors
        # Filter out rows where x or y are NaN (can't plot those)
        scatter_data = scatter_data.dropna(subset=["bsr", "price", "revenue_proxy"])
        scatter_data["monthly_units"] = scatter_data["monthly_units"].fillna(1)
        
        if scatter_data.empty:
            st.warning("⚠️ No valid data points for scatter plot (missing BSR, price, or revenue data)")
        else:
            fig = px.scatter(
                scatter_data,
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
            st.plotly_chart(fig, use_container_width=True, key="discovery_price_bsr_chart")

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
    render_pin_to_state_ui(market_snapshot, stats, context="discovery")


def render_pin_to_state_ui(market_snapshot: pd.DataFrame, stats: dict, context: str = "discovery") -> None:
    """
    UI for creating a new project from discovered ASINs.

    Args:
        market_snapshot: Pruned DataFrame from discovery
        stats: Discovery statistics
        context: Context identifier to make keys unique (default: "discovery")
    """
    st.markdown("### 📌 Pin to State (Create Project)")

    col1, col2 = st.columns([2, 1])

    with col1:
        project_name = st.text_input(
            "Project Name",
            value=f"{stats.get('query', 'Market')} Monitoring",
            placeholder="e.g., Starbucks K-Cups Q1 2026",
            key=f"pin_to_state_project_name_{context}"
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
            format_func=lambda x: mission_options[x],
            key=f"mission_profile_{context}"
        )

    # Show mission profile details
    config = get_mission_profile_config(mission_type)
    with st.expander(f"ℹ️ About {config['name']}"):
        st.markdown(f"**Focus**: {config['focus']}")
        st.markdown("**Top Priorities:**")
        top_3 = sorted(config["priorities"].items(), key=lambda x: x[1], reverse=True)[:3]
        for priority, weight in top_3:
            st.markdown(f"- {priority.replace('_', ' ').title()} (weight: {weight})")

    # Show success message if just saved
    if st.session_state.get("just_saved_mapping"):
        saved_name = st.session_state.get("saved_mapping_name", "Mapping")
        st.success(
            f"✅ Mapping '{saved_name}' saved to User Dashboard!\n\n"
            f"💡 Click the **User Dashboard** tab to view your saved mapping."
        )
        if st.button("🔄 Start New Search", key=f"new_search_after_save_{context}"):
            # Clear discovery state
            for key in ["discovery_market_snapshot", "discovery_stats", "seed_products_df", 
                       "last_search", "selected_seed_idx", "trigger_phase2", "just_saved_mapping"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return  # Don't show the form again after saving

    # Pin button
    if st.button("🚀 Create Project & Backfill History", type="primary", key=f"create_project_btn_{context}"):
        with st.spinner("📊 Processing market data..."):
            asins = market_snapshot["asin"].tolist()
            
            # Use df_weekly from market_stats if available (already fetched in phase2)
            # This avoids duplicate API calls since phase2 now fetches 90-day history
            df_weekly = stats.get("df_weekly") if isinstance(stats.get("df_weekly"), pd.DataFrame) else None
            
            if df_weekly is None or df_weekly.empty:
                # Fallback: Fetch detailed weekly data if not already in stats
                from src.two_phase_discovery import fetch_detailed_weekly_data
                st.caption("📊 Fetching 3 months of detailed market data...")
                df_weekly = fetch_detailed_weekly_data(tuple(asins), days=90)  # Tuple for caching
            else:
                st.caption(f"✅ Using pre-fetched 90-day historical data ({len(df_weekly)} rows)")
            
            # Save mapping with detailed data to User Dashboard
            mapping_id = save_market_mapping(
                project_name,
                market_snapshot,
                stats,
                df_weekly=df_weekly  # Include detailed weekly data
            )

            # === ACTIVATE COMMAND CENTER ===
            # Set active project ASIN to enable Command Center
            if asins:
                # Get seed product info from last Phase 2 params
                seed_params = st.session_state.get("last_phase2_params", {})
                seed_title = seed_params.get("seed_product_title", "")
                seed_asin = seed_params.get("seed_product_asin", asins[0])  # ✅ GET SEED ASIN

                # Extract brand from seed title (first word usually)
                seed_brand = seed_title.split()[0] if seed_title else ""

                st.session_state["active_project_asin"] = seed_asin  # ✅ USE SEED ASIN (not asins[0])
                st.session_state["active_project_data"] = df_weekly  # Store 90-day data
                st.session_state["active_project_name"] = project_name
                st.session_state["active_project_market_snapshot"] = market_snapshot  # Store full market
                st.session_state["active_project_seed_brand"] = seed_brand  # Track seed brand
                st.session_state["active_project_all_asins"] = asins  # All market ASINs

            # Mark as just saved so we show success on rerun
            st.session_state["just_saved_mapping"] = True
            st.session_state["saved_mapping_name"] = project_name

            # === CACHE TO SUPABASE FOR INSTANT RETURN VISITS ===
            # This is the key to making "search today, cached tomorrow" work
            # ENHANCEMENT 2.3: Consolidated caching - single write with all fields
            if CACHE_ENABLED and cache_market_snapshot:
                try:
                    # Get category context from last Phase 2 params
                    seed_params = st.session_state.get("last_phase2_params", {})
                    category_id = seed_params.get("category_id", 0)
                    category_path = seed_params.get("category_path", "")
                    category_name = category_path.split(' > ')[-1] if category_path else 'Unknown'
                    category_tree = category_path.split(' > ') if category_path else []
                    category_root = category_tree[0] if category_tree else category_name

                    # Build category context for consolidated write
                    category_context = {
                        "category_id": int(category_id) if category_id else None,
                        "category_name": category_name,
                        "category_tree": category_tree,
                        "category_root": category_root
                    }

                    # Step 1: Cache product snapshots WITH category metadata (consolidated)
                    cached_count = cache_market_snapshot(market_snapshot, df_weekly, category_context)

                    # Step 2: Accumulate network intelligence (category benchmarks, patterns)
                    # Skip product snapshot write since we already did it with category metadata
                    try:
                        from src.data_accumulation import NetworkIntelligenceAccumulator
                        from supabase import create_client
                        import os

                        # Get Supabase credentials
                        supabase_url = st.secrets.get("supabase", {}).get("url") or st.secrets.get("url") or os.getenv("SUPABASE_URL")
                        supabase_key = st.secrets.get("supabase", {}).get("service_key") or st.secrets.get("key") or os.getenv("SUPABASE_SERVICE_KEY")

                        if supabase_url and supabase_key:
                            supabase = create_client(supabase_url, supabase_key)
                            accumulator = NetworkIntelligenceAccumulator(supabase)

                            # Accumulate network intelligence (skip snapshot write - already done)
                            accumulator.accumulate_search_data(
                                market_snapshot=market_snapshot.copy(),
                                category_id=int(category_id) if category_id else 0,
                                category_name=category_name,
                                category_tree=category_tree,
                                skip_snapshot_write=True  # ENHANCEMENT 2.3: Avoid duplicate write
                            )

                            if cached_count > 0:
                                st.caption(f"⚡ Cached {cached_count} products + network intelligence for instant future loads")
                        else:
                            if cached_count > 0:
                                st.caption(f"⚡ Cached {cached_count} products for instant future loads")
                    except Exception as e:
                        # Network accumulation failed, but basic cache worked
                        if cached_count > 0:
                            st.caption(f"⚡ Cached {cached_count} products for instant future loads")
                except Exception as e:
                    pass  # Caching failed, not critical

            # Try to persist to Supabase (optional - may fail if tables don't exist)
            try:
                project_id = pin_to_state(
                    asins=asins,
                    project_name=project_name,
                    mission_type=mission_type,
                    user_id=None,
                    metadata={
                        **stats,
                        "search_date": pd.Timestamp.now().isoformat()
                    }
                )

                # Trigger async backfill to Supabase
                execute_backfill(
                    project_id=project_id,
                    asins=asins,
                    run_async=True
                )

                st.session_state["active_project_id"] = project_id
                st.session_state["show_project_dashboard"] = True

            except Exception as e:
                pass  # Supabase failed, but mapping is saved to session state

        # Show success message with Command Center activation
        st.success(
            f"✅ Project '{project_name}' created!\n\n"
            f"🛡️ **Command Center Activated** - Navigate to the Command Center tab to view your defense perimeter."
        )

        # Rerun to activate Command Center
        st.rerun()


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
            st.plotly_chart(fig, use_container_width=True, key="project_dashboard_bsr_trend")

        # Price trend
        if "buy_box_price" in df_metrics.columns:
            fig = px.line(
                df_metrics.groupby("datetime")["buy_box_price"].mean().reset_index(),
                x="datetime",
                y="buy_box_price",
                title="Average Buy Box Price",
                labels={"datetime": "Date", "buy_box_price": "Price ($)"}
            )
            st.plotly_chart(fig, use_container_width=True, key="project_dashboard_price_trend")

    except Exception as e:
        st.error(f"❌ Failed to load project dashboard: {str(e)}")


def render_project_selector() -> Optional[str]:
    """
    Main area component for selecting existing projects.

    Returns:
        Selected project_id or None
    """
    st.markdown("### 📂 Your Projects")

    # Load user projects
    projects_df = load_user_projects(user_id=None)  # TODO: Get from auth

    if projects_df.empty:
        st.info("No projects yet. Use Discovery to create one.")
        return None

    # Project dropdown
    project_options = {
        row["id"]: f"{row['project_name']} ({row['asin_count']} ASINs)"
        for _, row in projects_df.iterrows()
    }

    selected_id = st.selectbox(
        "Select Project",
        options=list(project_options.keys()),
        format_func=lambda x: project_options[x],
        label_visibility="visible"
    )

    return selected_id


def save_market_mapping(
    mapping_name: str, 
    market_snapshot: pd.DataFrame, 
    stats: dict,
    df_weekly: pd.DataFrame = None
) -> str:
    """
    Save a market mapping to session state for User Dashboard.
    
    Args:
        mapping_name: User-provided name for the mapping
        market_snapshot: DataFrame with market data (Phase 2 results)
        stats: Discovery statistics
        df_weekly: Detailed weekly data for analysis (3 months)
        
    Returns:
        mapping_id: Unique identifier for the mapping
    """
    import uuid
    from datetime import datetime
    
    # Initialize mappings storage if not exists
    if "user_mappings" not in st.session_state:
        st.session_state["user_mappings"] = {}
    
    mapping_id = str(uuid.uuid4())
    
    st.session_state["user_mappings"][mapping_id] = {
        "id": mapping_id,
        "name": mapping_name,
        "created_at": datetime.utcnow().isoformat(),
        "market_snapshot": market_snapshot,
        "stats": stats,
        "df_weekly": df_weekly if df_weekly is not None else pd.DataFrame()
    }
    
    return mapping_id


def load_user_mappings() -> dict:
    """
    Load all saved market mappings from session state.
    
    Returns:
        Dict of mapping_id -> mapping data
    """
    return st.session_state.get("user_mappings", {})


def delete_user_mapping(mapping_id: str) -> bool:
    """
    Delete a market mapping from session state.
    
    Args:
        mapping_id: UUID of the mapping to delete
        
    Returns:
        Success boolean
    """
    if "user_mappings" in st.session_state and mapping_id in st.session_state["user_mappings"]:
        del st.session_state["user_mappings"][mapping_id]
        return True
    return False


def render_save_mapping_ui(market_snapshot: pd.DataFrame, stats: dict) -> None:
    """
    UI for saving a market mapping to User Dashboard.
    
    Args:
        market_snapshot: DataFrame from discovery
        stats: Discovery statistics
    """
    st.markdown("### 💾 Save Market Mapping")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        mapping_name = st.text_input(
            "Mapping Name",
            value=stats.get('query', 'Market Mapping').replace('"', ''),
            placeholder="e.g., Greek Yogurt Market Q1 2026",
            key="save_mapping_name"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("💾 Save to Dashboard", type="primary", use_container_width=True):
            mapping_id = save_market_mapping(mapping_name, market_snapshot, stats)
            st.success(f"✅ Mapping '{mapping_name}' saved! View in **User Dashboard** tab.")
            st.session_state["last_saved_mapping_id"] = mapping_id


def render_user_dashboard() -> None:
    """
    User Dashboard - displays saved market mappings with full analysis.
    
    Uses the same layout and analysis engine as Current Dashboard:
    - Executive briefing with AI insights
    - Strategic metrics tiles
    - Problem category breakdown
    - Visualizations (Market Share, Price vs BSR, Top ASINs)
    """
    st.markdown("## 👤 User Dashboard")
    st.markdown("Your saved market mappings and competitive intelligence")
    
    mappings = load_user_mappings()
    
    if not mappings:
        st.info(
            "💡 No saved mappings yet.\n\n"
            "Use the **Market Discovery** tab to find and map a competitive market, "
            "then save it to view here."
        )
        return
    
    # Mapping selector
    mapping_options = {
        mid: f"{m['name']} ({m['stats'].get('total_asins', 0)} ASINs)"
        for mid, m in mappings.items()
    }
    
    selected_mapping_id = st.selectbox(
        "Select Mapping",
        options=list(mapping_options.keys()),
        format_func=lambda x: mapping_options[x],
        key="user_dashboard_mapping_selector"
    )
    
    if not selected_mapping_id:
        return
    
    mapping = mappings[selected_mapping_id]
    market_snapshot = mapping["market_snapshot"]
    stats = mapping["stats"]
    df_weekly = mapping.get("df_weekly", pd.DataFrame())
    created_at = mapping.get("created_at", "Unknown")
    
    # Header with delete button
    col_header, col_delete = st.columns([4, 1])
    with col_header:
        st.markdown(f"### 📊 {mapping['name']}")
        st.caption(f"Created: {created_at[:10] if len(created_at) >= 10 else created_at} | {stats.get('total_asins', 0)} ASINs tracked")
    with col_delete:
        if st.button("🗑️ Delete", key=f"delete_{selected_mapping_id}"):
            delete_user_mapping(selected_mapping_id)
            st.rerun()
    
    # Check if we have detailed weekly data for analysis
    if df_weekly is not None and not df_weekly.empty:
        try:
            # Ensure datetime format
            df_weekly["week_start"] = pd.to_datetime(df_weekly["week_start"], utc=True)
            all_weeks = sorted(df_weekly["week_start"].dt.date.unique(), reverse=True)
            
            if all_weeks:
                selected_week = all_weeks[0]
                
                # Get most recent week's data
                df_current = df_weekly[df_weekly["week_start"].dt.date == selected_week].copy()
                
                if not df_current.empty:
                    # Calculate basic metrics from the weekly data
                    total_rev_curr = df_current["weekly_sales_filled"].sum() if "weekly_sales_filled" in df_current.columns else 0
                    total_units = df_current["estimated_units"].sum() if "estimated_units" in df_current.columns else 0
                    avg_price = df_current["filled_price"].mean() if "filled_price" in df_current.columns else 0
                    avg_bsr = df_current["sales_rank_filled"].mean() if "sales_rank_filled" in df_current.columns else 0
                    num_products = df_current["asin"].nunique()
                    
                    # Simple status based on revenue
                    status_emoji, status_text, status_color = "🟢", "TRACKING", "#28a745"
                    
                    # Format money helper
                    def fmt_money(val):
                        if val >= 1_000_000:
                            return f"${val/1_000_000:.1f}M"
                        elif val >= 1_000:
                            return f"${val/1_000:.0f}K"
                        else:
                            return f"${val:.0f}"
                    
                    # Render executive briefing
                    st.markdown("---")
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; 
                                margin-bottom: 20px; border-left: 5px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                            <div>
                                <div style="font-size: 12px; color: #666; margin-bottom: 4px;">Week of {selected_week}</div>
                                <div style="font-size: 22px; font-weight: 700; color: #1a1a1a;">
                                    {status_emoji} {status_text}: {num_products} products monitored
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 28px; font-weight: 700; color: #00704A;">{fmt_money(total_rev_curr)}</div>
                                <div style="font-size: 11px; color: #666;">Weekly Revenue (Est.)</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Strategic metrics tiles
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.metric("📈 Weekly Revenue", fmt_money(total_rev_curr))
                    with c2:
                        st.metric("📦 Est. Weekly Units", f"{total_units:,.0f}")
                    with c3:
                        st.metric("💰 Avg Price", f"${avg_price:.2f}" if avg_price > 0 else "N/A")
                    with c4:
                        st.metric("📊 Avg BSR", f"{avg_bsr:,.0f}" if avg_bsr > 0 else "N/A")
                    
                    # Historical trend (if multiple weeks available)
                    if len(all_weeks) > 1:
                        st.markdown("---")
                        st.markdown("### 📈 Weekly Revenue Trend")
                        
                        weekly_rev = df_weekly.groupby(df_weekly["week_start"].dt.date)["weekly_sales_filled"].sum().reset_index()
                        weekly_rev.columns = ["Week", "Revenue"]
                        weekly_rev = weekly_rev.sort_values("Week")
                        
                        fig = px.line(
                            weekly_rev,
                            x="Week",
                            y="Revenue",
                            title="Weekly Revenue Over Time",
                            labels={"Week": "Week", "Revenue": "Revenue ($)"}
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True, key="user_dashboard_trend_chart")
                else:
                    st.info("📅 No data found for the most recent week")
        except Exception as e:
            st.warning(f"⚠️ Could not analyze detailed data: {str(e)}")
    
    # === VISUALIZATION TABS (Always show) ===
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Market Share", "💰 Price vs BSR", "📋 Top 20 ASINs"])
    
    with tab1:
        if not market_snapshot.empty and "revenue_proxy" in market_snapshot.columns:
            fig = px.pie(
                market_snapshot.head(20),
                values="revenue_proxy",
                names="title" if "title" in market_snapshot.columns else "asin",
                title="Revenue Distribution (Top 20 ASINs)",
                hole=0.4
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="user_dashboard_market_share_chart")
        else:
            st.info("No revenue data available for visualization.")
    
    with tab2:
        if not market_snapshot.empty and "bsr" in market_snapshot.columns and "price" in market_snapshot.columns:
            fig = px.scatter(
                market_snapshot,
                x="bsr",
                y="price",
                size="monthly_units" if "monthly_units" in market_snapshot.columns else None,
                color="revenue_proxy" if "revenue_proxy" in market_snapshot.columns else None,
                hover_data=["title", "asin"] if "title" in market_snapshot.columns else ["asin"],
                title="Price vs Sales Rank (BSR)",
                labels={
                    "bsr": "Sales Rank (BSR)",
                    "price": "Price ($)",
                    "monthly_units": "Monthly Units",
                    "revenue_proxy": "Revenue"
                }
            )
            fig.update_layout(height=500)
            fig.update_xaxes(type="log")
            st.plotly_chart(fig, use_container_width=True, key="user_dashboard_price_bsr_chart")
        else:
            st.info("No price/BSR data available for visualization.")
    
    with tab3:
        if not market_snapshot.empty:
            display_cols = [col for col in ["asin", "title", "price", "bsr", "monthly_units", "revenue_proxy"] 
                          if col in market_snapshot.columns]
            
            if display_cols:
                top_20 = market_snapshot.head(20)[display_cols].copy()
                
                if "price" in top_20.columns:
                    top_20["price"] = top_20["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                if "bsr" in top_20.columns:
                    top_20["bsr"] = top_20["bsr"].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "N/A")
                if "monthly_units" in top_20.columns:
                    top_20["monthly_units"] = top_20["monthly_units"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
                if "revenue_proxy" in top_20.columns:
                    top_20["revenue_proxy"] = top_20["revenue_proxy"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                if "title" in top_20.columns:
                    top_20["title"] = top_20["title"].apply(lambda x: str(x)[:60] + "..." if len(str(x)) > 60 else x)
                
                column_names = {
                    "asin": "ASIN", "title": "Title", "price": "Price",
                    "bsr": "BSR", "monthly_units": "Monthly Units", "revenue_proxy": "Revenue (Est.)"
                }
                top_20 = top_20.rename(columns=column_names)
                st.dataframe(top_20, use_container_width=True, hide_index=True)
            else:
                st.info("No data available to display.")
        else:
            st.info("No ASINs in this mapping.")
