"""
ShelfGuard LLM Engine Debug Dashboard
======================================
Interactive testing and validation of the AI Strategic Triangulator.

Test Areas:
1. Data Healer - Gap filling and interpolation quality
2. LLM Classifier - Strategic state recommendations
3. Competitive Intelligence - Signal extraction
4. Fallback Logic - Deterministic vs LLM comparison
5. Performance - Response times and token usage
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time
import json

# Import the AI engine and data healer
try:
    from utils.ai_engine import StrategicTriangulator, StrategicState, STATE_DEFINITIONS
    from utils.data_healer import (
        clean_and_interpolate_metrics,
        generate_data_quality_report,
        validate_healing,
    )
    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    st.error(f"AI Engine not available: {e}")


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_test_product(scenario: str = "fortress") -> dict:
    """Generate test product data for different scenarios."""
    
    base_data = {
        "asin": f"TEST_{scenario.upper()}",
        "product_name": f"Test Product - {scenario.title()}",
        "week_start": datetime.now().strftime("%Y-W%V"),
    }
    
    scenarios = {
        "fortress": {
            "filled_price": 24.99,
            "sales_rank": 850,
            "rank_delta_90d_pct": -5,  # Improving
            "new_offer_count": 3,
            "delta30_COUNT_NEW": 0,
            "amazon_bb_share": 0.92,
            "review_count": 450,
            "delta30_COUNT_REVIEWS": 12,
            "rating": 4.7,
            "net_margin": 0.22,
            "velocity_decay": 0.98,
        },
        "harvest": {
            "filled_price": 29.99,
            "sales_rank": 1250,
            "rank_delta_90d_pct": 2,  # Stable
            "new_offer_count": 5,
            "delta30_COUNT_NEW": 0,
            "amazon_bb_share": 0.88,
            "review_count": 890,
            "delta30_COUNT_REVIEWS": 3,
            "rating": 4.6,
            "net_margin": 0.28,
            "velocity_decay": 1.02,
        },
        "trench_war": {
            "filled_price": 19.99,
            "sales_rank": 2100,
            "rank_delta_90d_pct": 15,  # Declining
            "new_offer_count": 15,
            "delta30_COUNT_NEW": 7,  # +7 new sellers!
            "amazon_bb_share": 0.58,
            "review_count": 320,
            "delta30_COUNT_REVIEWS": 8,
            "rating": 4.4,
            "net_margin": 0.14,
            "velocity_decay": 1.12,
        },
        "distress": {
            "filled_price": 16.99,
            "sales_rank": 4500,
            "rank_delta_90d_pct": 25,  # Declining fast
            "new_offer_count": 12,
            "delta30_COUNT_NEW": 3,
            "amazon_bb_share": 0.65,
            "review_count": 180,
            "delta30_COUNT_REVIEWS": 1,
            "rating": 4.2,
            "net_margin": 0.08,
            "velocity_decay": 1.18,
        },
        "terminal": {
            "filled_price": 12.99,
            "sales_rank": 12000,
            "rank_delta_90d_pct": 45,  # Severe decline
            "new_offer_count": 20,
            "delta30_COUNT_NEW": 5,
            "amazon_bb_share": 0.45,
            "review_count": 95,
            "delta30_COUNT_REVIEWS": 0,
            "rating": 3.9,
            "net_margin": -0.03,
            "velocity_decay": 1.35,
        },
    }
    
    if scenario in scenarios:
        base_data.update(scenarios[scenario])
    
    return base_data


def generate_test_dataset_with_gaps(num_products: int = 10) -> pd.DataFrame:
    """Generate a test dataset with intentional gaps for healer testing."""
    
    data = []
    scenarios = ["fortress", "harvest", "trench_war", "distress", "terminal"]
    
    for i in range(num_products):
        scenario = scenarios[i % len(scenarios)]
        product = generate_test_product(scenario)
        product["asin"] = f"TEST{i:03d}"
        
        # Introduce random gaps (20% probability per field)
        for key in ["filled_price", "new_offer_count", "rating", "review_count", "delta30_COUNT_NEW"]:
            if np.random.random() < 0.2:
                product[key] = np.nan
        
        data.append(product)
    
    return pd.DataFrame(data)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_data_healer(st_container):
    """Test the Universal Data Healer."""
    st_container.subheader("🩹 Data Healer Test")
    
    # Generate test data with gaps
    df_test = generate_test_dataset_with_gaps(10)
    
    # Show before stats
    gaps_before = df_test.isna().sum().sum()
    st_container.metric("Gaps Before Healing", gaps_before)
    
    col1, col2 = st_container.columns(2)
    
    with col1:
        st.write("**Sample Data (Before):**")
        st.dataframe(
            df_test[["asin", "filled_price", "new_offer_count", "rating"]].head(5),
            use_container_width=True
        )
    
    # Apply healer
    start_time = time.time()
    df_healed = clean_and_interpolate_metrics(df_test, group_by_column="asin", verbose=False)
    heal_time = time.time() - start_time
    
    # Show after stats
    gaps_after = df_healed.isna().sum().sum()
    
    with col2:
        st.write("**Sample Data (After):**")
        st.dataframe(
            df_healed[["asin", "filled_price", "new_offer_count", "rating"]].head(5),
            use_container_width=True
        )
    
    # Metrics
    col1, col2, col3 = st_container.columns(3)
    col1.metric("Gaps After Healing", gaps_after, delta=f"-{gaps_before - gaps_after}")
    col2.metric("Healing Time", f"{heal_time:.3f}s")
    col3.metric("Success Rate", f"{100 if gaps_after == 0 else 0}%")
    
    # Quality report
    quality_report = generate_data_quality_report(df_healed, group_by="asin")
    
    is_valid, issues = validate_healing(df_healed)
    
    if is_valid:
        st_container.success("✓ Validation Passed: All critical columns healed")
    else:
        st_container.error(f"✗ Validation Failed: {issues}")
    
    return df_healed


def test_llm_classifier(st_container, triangulator, test_scenario):
    """Test the LLM classifier with a specific scenario."""
    st_container.subheader(f"🤖 LLM Classifier Test: {test_scenario.upper()}")
    
    # Generate test product
    product_data = generate_test_product(test_scenario)
    
    # Show input data
    with st_container.expander("📊 Input Data", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Pricing & Performance**")
            st.write(f"Price: ${product_data['filled_price']}")
            st.write(f"Sales Rank: {product_data['sales_rank']:,}")
            st.write(f"Rank Δ 90d: {product_data['rank_delta_90d_pct']:+.0f}%")
            st.write(f"Margin: {product_data['net_margin']*100:.0f}%")
        
        with col2:
            st.write("**Competition**")
            st.write(f"Seller Count: {product_data['new_offer_count']}")
            st.write(f"Sellers Δ 30d: {product_data['delta30_COUNT_NEW']:+.0f}")
            st.write(f"Buy Box: {product_data['amazon_bb_share']*100:.0f}%")
        
        with col3:
            st.write("**Social Proof**")
            st.write(f"Reviews: {product_data['review_count']:,}")
            st.write(f"Reviews Δ 30d: {product_data['delta30_COUNT_REVIEWS']:+.0f}")
            st.write(f"Rating: {product_data['rating']:.1f} ⭐")
    
    # Test both LLM and fallback
    col1, col2 = st_container.columns(2)
    
    # LLM Mode
    with col1:
        st.write("**🤖 LLM Mode**")
        try:
            start_time = time.time()
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            brief_llm = loop.run_until_complete(
                triangulator.analyze_strategy_with_llm(product_data)
            )
            loop.close()
            
            llm_time = time.time() - start_time
            
            st.success(f"State: {brief_llm.state_emoji} **{brief_llm.strategic_state}**")
            st.metric("Confidence", f"{brief_llm.confidence_score*100:.0f}%")
            st.metric("Response Time", f"{llm_time:.2f}s")
            st.write(f"**Reasoning:**")
            st.info(brief_llm.reasoning)
            st.write(f"**Recommended Action:**")
            st.warning(brief_llm.recommended_action)
            
            # Show signals
            if brief_llm.signals_detected:
                st.write("**Signals Detected:**")
                for signal in brief_llm.signals_detected:
                    st.caption(f"• {signal}")
            
        except Exception as e:
            st.error(f"LLM Error: {str(e)}")
            brief_llm = None
    
    # Fallback Mode
    with col2:
        st.write("**🔧 Fallback Mode (Deterministic)**")
        try:
            start_time = time.time()
            brief_fallback = triangulator._determine_state_fallback(product_data)
            fallback_time = time.time() - start_time
            
            st.info(f"State: {brief_fallback.state_emoji} **{brief_fallback.strategic_state}**")
            st.metric("Confidence", f"{brief_fallback.confidence_score*100:.0f}%")
            st.metric("Response Time", f"{fallback_time:.3f}s")
            st.write(f"**Reasoning:**")
            st.caption(brief_fallback.reasoning or "Deterministic classification")
            st.write(f"**Recommended Action:**")
            st.caption(brief_fallback.recommended_action or "See plan")
            
            # Show signals
            if brief_fallback.signals_detected:
                st.write("**Signals Detected:**")
                for signal in brief_fallback.signals_detected:
                    st.caption(f"• {signal}")
        
        except Exception as e:
            st.error(f"Fallback Error: {str(e)}")
            brief_fallback = None
    
    # Comparison
    if brief_llm and brief_fallback:
        st_container.write("---")
        st_container.write("**🔍 Comparison**")
        
        comparison_data = {
            "Metric": ["Strategic State", "Confidence", "Response Time", "Reasoning Quality"],
            "LLM": [
                brief_llm.strategic_state,
                f"{brief_llm.confidence_score*100:.0f}%",
                f"{llm_time:.2f}s",
                f"{len(brief_llm.reasoning)} chars"
            ],
            "Fallback": [
                brief_fallback.strategic_state,
                f"{brief_fallback.confidence_score*100:.0f}%",
                f"{fallback_time:.3f}s",
                f"{len(brief_fallback.reasoning) if brief_fallback.reasoning else 0} chars"
            ],
            "Match": [
                "✓" if brief_llm.strategic_state == brief_fallback.strategic_state else "✗",
                "✓" if abs(brief_llm.confidence_score - brief_fallback.confidence_score) < 0.1 else "✗",
                "-",
                "-"
            ]
        }
        
        st_container.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    return brief_llm, brief_fallback


def test_batch_performance(st_container, triangulator, df_test):
    """Test batch processing performance."""
    st_container.subheader("⚡ Batch Performance Test")
    
    num_products = len(df_test)
    st_container.write(f"Testing with {num_products} products...")
    
    # Run batch analysis
    try:
        start_time = time.time()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            triangulator.triangulate_portfolio(df_test)
        )
        loop.close()
        
        batch_time = time.time() - start_time
        
        # Metrics
        col1, col2, col3, col4 = st_container.columns(4)
        col1.metric("Total Time", f"{batch_time:.2f}s")
        col2.metric("Per Product", f"{batch_time/num_products:.3f}s")
        col3.metric("Products Classified", len(results))
        col4.metric("Success Rate", f"{len(results)/num_products*100:.0f}%")
        
        # State distribution
        states = {}
        confidences = []
        
        for result in results:
            state = result.get("strategic_state", "UNKNOWN")
            states[state] = states.get(state, 0) + 1
            confidences.append(result.get("confidence_score", 0))
        
        # Show distribution
        col1, col2 = st_container.columns(2)
        
        with col1:
            st.write("**State Distribution**")
            state_df = pd.DataFrame({
                "State": list(states.keys()),
                "Count": list(states.values()),
                "Percentage": [f"{v/len(results)*100:.0f}%" for v in states.values()]
            })
            st.dataframe(state_df, use_container_width=True)
        
        with col2:
            st.write("**Confidence Statistics**")
            st.metric("Average Confidence", f"{np.mean(confidences)*100:.0f}%")
            st.metric("Min Confidence", f"{np.min(confidences)*100:.0f}%")
            st.metric("Max Confidence", f"{np.max(confidences)*100:.0f}%")
            st.metric("High Conf (>85%)", f"{sum(c > 0.85 for c in confidences)}/{len(confidences)}")
        
        # Show sample results
        with st_container.expander("📋 Sample Results", expanded=False):
            results_df = pd.DataFrame(results)
            st.dataframe(
                results_df[["asin", "strategic_state", "confidence_score", "recommended_action"]].head(10),
                use_container_width=True
            )
        
        return results
        
    except Exception as e:
        st_container.error(f"Batch processing error: {str(e)}")
        return None


def test_competitive_intelligence(st_container):
    """Test competitive intelligence signal extraction."""
    st_container.subheader("🎯 Competitive Intelligence Signals")
    
    # Test each scenario
    scenarios = ["fortress", "harvest", "trench_war", "distress", "terminal"]
    
    data = []
    for scenario in scenarios:
        product = generate_test_product(scenario)
        data.append({
            "Scenario": scenario.title(),
            "Seller Count": product["new_offer_count"],
            "Seller Δ 30d": f"{product['delta30_COUNT_NEW']:+.0f}",
            "Buy Box %": f"{product['amazon_bb_share']*100:.0f}%",
            "Margin": f"{product['net_margin']*100:.0f}%",
            "Rank Δ 90d": f"{product['rank_delta_90d_pct']:+.0f}%",
            "Expected State": scenario.upper(),
        })
    
    st_container.dataframe(pd.DataFrame(data), use_container_width=True)
    
    st_container.write("**Signal Interpretation:**")
    st_container.caption("• FORTRESS: Low competition, high Buy Box %, strong margin")
    st_container.caption("• HARVEST: Stable metrics, premium pricing, good margin")
    st_container.caption("• TRENCH_WAR: High competition, increasing sellers, declining Buy Box")
    st_container.caption("• DISTRESS: Margin compression, rank decay, competitive pressure")
    st_container.caption("• TERMINAL: Negative margin, severe rank decline, high competition")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="LLM Engine Debug",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 LLM Engine Debug Dashboard")
    st.caption("Test and validate the AI Strategic Triangulator")
    
    if not ENGINE_AVAILABLE:
        st.error("AI Engine not available. Check imports.")
        return
    
    # Sidebar controls
    st.sidebar.header("Test Configuration")
    
    use_llm = st.sidebar.checkbox("Enable LLM Mode", value=True, 
                                   help="If unchecked, only fallback logic will be used")
    
    test_mode = st.sidebar.selectbox(
        "Select Test",
        [
            "1. Data Healer Test",
            "2. Single Product Test",
            "3. Competitive Intelligence",
            "4. Batch Performance Test",
            "5. All Tests"
        ]
    )
    
    # Initialize triangulator
    try:
        triangulator = StrategicTriangulator(use_llm=use_llm)
        st.sidebar.success(f"✓ Triangulator initialized ({'LLM' if use_llm else 'Fallback'} mode)")
    except Exception as e:
        st.error(f"Failed to initialize triangulator: {e}")
        return
    
    # Run selected test
    st.write("---")
    
    if "1. Data Healer" in test_mode or "5. All Tests" in test_mode:
        df_healed = test_data_healer(st.container())
        st.write("---")
    else:
        df_healed = generate_test_dataset_with_gaps(10)
        df_healed = clean_and_interpolate_metrics(df_healed, group_by_column="asin")
    
    if "2. Single Product" in test_mode or "5. All Tests" in test_mode:
        scenario = st.sidebar.selectbox(
            "Select Scenario",
            ["fortress", "harvest", "trench_war", "distress", "terminal"]
        )
        test_llm_classifier(st.container(), triangulator, scenario)
        st.write("---")
    
    if "3. Competitive Intelligence" in test_mode or "5. All Tests" in test_mode:
        test_competitive_intelligence(st.container())
        st.write("---")
    
    if "4. Batch Performance" in test_mode or "5. All Tests" in test_mode:
        test_batch_performance(st.container(), triangulator, df_healed)
        st.write("---")
    
    # System info
    st.sidebar.write("---")
    st.sidebar.subheader("System Info")
    st.sidebar.caption(f"OpenAI Client: {'✓' if hasattr(triangulator, 'openai_client') else '✗'}")
    st.sidebar.caption(f"Data Healer: ✓")
    st.sidebar.caption(f"State Definitions: {len(STATE_DEFINITIONS)}")
    
    # Export results
    if st.sidebar.button("Export Test Results"):
        st.sidebar.download_button(
            label="Download as JSON",
            data=json.dumps({
                "timestamp": datetime.now().isoformat(),
                "mode": "LLM" if use_llm else "Fallback",
                "test_completed": test_mode,
            }, indent=2),
            file_name=f"llm_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
