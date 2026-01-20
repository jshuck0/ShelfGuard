"""
Example: How to Use the Intelligence Pipeline with Phase 2 Discovery

This shows the complete integration flow:
1. Run Phase 1 (seed discovery)
2. Run Phase 2 (market mapping)
3. Run Intelligence Pipeline (generate insights)
4. Display results
"""

import streamlit as st
import pandas as pd
from src.two_phase_discovery import (
    phase1_seed_discovery,
    phase2_category_market_mapping,
    generate_strategic_intelligence
)

# ========================================
# EXAMPLE USAGE
# ========================================

def example_full_discovery_with_intelligence():
    """
    Complete example: Search → Discover → Analyze → Generate Insights
    """

    st.title("🧠 ShelfGuard Intelligence Pipeline Demo")

    # ========================================
    # STEP 1: PHASE 1 - SEED DISCOVERY
    # ========================================

    st.header("1️⃣ Phase 1: Seed Discovery")

    keyword = st.text_input("Search keyword", value="k-cup coffee")

    if st.button("🔍 Search Products"):
        with st.spinner("Searching Amazon..."):
            df_seeds = phase1_seed_discovery(
                keyword=keyword,
                limit=50,
                domain="US"
            )

        if not df_seeds.empty:
            st.success(f"Found {len(df_seeds)} products")

            # Let user select seed product
            seed_options = df_seeds.apply(
                lambda row: f"{row['title'][:50]}... (BSR: {row['bsr']:,})",
                axis=1
            ).tolist()

            selected_idx = st.selectbox(
                "Select seed product to define market",
                range(len(seed_options)),
                format_func=lambda x: seed_options[x]
            )

            seed_row = df_seeds.iloc[selected_idx]
            seed_asin = seed_row['asin']
            seed_title = seed_row['title']
            category_id = seed_row['category_id']
            leaf_category_id = seed_row.get('leaf_category_id')
            category_path = seed_row.get('category_path', '')

            st.session_state['seed_asin'] = seed_asin
            st.session_state['seed_title'] = seed_title
            st.session_state['category_id'] = category_id
            st.session_state['leaf_category_id'] = leaf_category_id
            st.session_state['category_path'] = category_path

            st.info(f"✅ Selected: {seed_title}")
            st.caption(f"Category: {category_path}")

    # ========================================
    # STEP 2: PHASE 2 - MARKET MAPPING
    # ========================================

    if 'seed_asin' in st.session_state:
        st.header("2️⃣ Phase 2: Market Mapping")

        st.info(f"Seed: {st.session_state['seed_title'][:60]}...")
        st.caption(f"Category: {st.session_state['category_path']}")

        if st.button("🗺️ Map Market"):
            with st.spinner("Fetching 100 ASINs from category..."):
                df_market, market_stats = phase2_category_market_mapping(
                    category_id=st.session_state['category_id'],
                    seed_product_title=st.session_state['seed_title'],
                    seed_asin=st.session_state['seed_asin'],
                    leaf_category_id=st.session_state.get('leaf_category_id'),
                    category_path=st.session_state.get('category_path'),
                    target_revenue_pct=80.0,
                    max_products=500,
                    batch_size=100,
                    domain="US"
                )

            if not df_market.empty:
                st.success(f"✅ Discovered {len(df_market)} products")
                st.metric(
                    "Total Market Revenue (Monthly)",
                    f"${market_stats.get('total_category_revenue', 0):,.0f}"
                )

                # Save to session state
                st.session_state['df_market'] = df_market
                st.session_state['df_weekly'] = market_stats.get('df_weekly', None)
                st.session_state['market_stats'] = market_stats

                # Show top products
                with st.expander("📊 Top 10 Products by Revenue"):
                    st.dataframe(
                        df_market[['title', 'brand', 'price', 'revenue_proxy', 'bsr']].head(10),
                        use_container_width=True
                    )

    # ========================================
    # STEP 3: INTELLIGENCE PIPELINE
    # ========================================

    if 'df_market' in st.session_state:
        st.header("3️⃣ Strategic Intelligence Generation")

        st.info("🧠 Ready to generate AI-powered insights")

        # Let user specify which ASINs to analyze
        portfolio_input = st.text_area(
            "Enter ASINs to analyze (one per line)",
            value=st.session_state.get('seed_asin', ''),
            help="These are YOUR products. Intelligence will be generated for these ASINs."
        )

        portfolio_asins = [
            asin.strip()
            for asin in portfolio_input.split('\n')
            if asin.strip()
        ]

        enable_network = st.checkbox(
            "Enable network intelligence accumulation",
            value=True,
            help="Stores market data in Supabase for network effect (category benchmarks, patterns)"
        )

        if st.button("💡 Generate Insights"):
            if not portfolio_asins:
                st.error("Please enter at least one ASIN to analyze")
            else:
                # Prepare category context
                category_context = {
                    'category_id': st.session_state['category_id'],
                    'category_name': st.session_state['category_path'].split(' > ')[-1] if st.session_state.get('category_path') else 'Unknown',
                    'category_path': st.session_state.get('category_path', '')
                }

                # Generate intelligence
                intelligence_results = generate_strategic_intelligence(
                    df_market_snapshot=st.session_state['df_market'],
                    df_weekly=st.session_state.get('df_weekly', pd.DataFrame()),
                    portfolio_asins=portfolio_asins,
                    category_context=category_context,
                    enable_network_accumulation=enable_network
                )

                # Save results
                st.session_state['intelligence_results'] = intelligence_results

    # ========================================
    # STEP 4: DISPLAY INSIGHTS
    # ========================================

    if 'intelligence_results' in st.session_state:
        st.header("4️⃣ Generated Insights")

        results = st.session_state['intelligence_results']

        if not results:
            st.warning("No insights generated")
        else:
            st.success(f"✅ Generated {len(results)} insights")

            # Group by priority
            critical = [r for r in results if r.product_status.priority == 100]
            opportunities = [r for r in results if r.product_status.priority == 75]
            watch = [r for r in results if r.product_status.priority == 50]
            stable = [r for r in results if r.product_status.priority == 0]

            # Display critical alerts
            if critical:
                st.error(f"🚨 CRITICAL ALERTS ({len(critical)})")
                for insight in critical:
                    display_insight_card(insight)

            # Display opportunities
            if opportunities:
                st.info(f"💡 OPPORTUNITIES ({len(opportunities)})")
                for insight in opportunities:
                    display_insight_card(insight)

            # Display watch items (collapsed)
            if watch:
                with st.expander(f"👀 WATCH ({len(watch)})", expanded=False):
                    for insight in watch:
                        display_insight_card(insight)

            # Display stable (collapsed)
            if stable:
                with st.expander(f"✅ STABLE ({len(stable)})", expanded=False):
                    for insight in stable:
                        display_insight_card(insight)


def display_insight_card(insight):
    """Display a single insight in a card format."""

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"**{insight.asin}** - {insight.strategic_state}")
            st.caption(insight.recommendation)

        with col2:
            upside = insight.projected_upside_monthly
            st.metric("Upside", f"${upside:.0f}/mo", delta=f"+{insight.net_expected_value:.0f}")

        with col3:
            downside = insight.downside_risk_monthly
            st.metric("Risk", f"${downside:.0f}/mo", delta=f"{insight.confidence:.0f}% conf")

        # Show details on expand
        if st.button(f"Details", key=f"details_{insight.asin}_{insight.timestamp}"):
            st.markdown("**AI Reasoning:**")
            st.info(insight.reasoning)

            st.markdown("**Trigger Events:**")
            if insight.trigger_events:
                for trigger in insight.trigger_events[:5]:
                    severity_emoji = "🔴" if trigger.severity >= 8 else "🟡" if trigger.severity >= 6 else "🟢"
                    st.markdown(
                        f"{severity_emoji} **{trigger.event_type}** (Severity: {trigger.severity}/10)  \n"
                        f"   {trigger.metric_name}: {trigger.baseline_value:.2f} → {trigger.current_value:.2f} "
                        f"({trigger.delta_pct:+.1f}%)"
                    )
            else:
                st.caption("No trigger events detected")

            st.markdown(f"**Action:** {insight.action_type} within {insight.time_horizon_days} days")

        st.divider()


# ========================================
# RUN EXAMPLE
# ========================================

if __name__ == "__main__":
    example_full_discovery_with_intelligence()
