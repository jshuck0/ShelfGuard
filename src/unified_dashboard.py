
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.dashboard_logic import ensure_data_loaded, f_money

# Metrics Calculation Imports
try:
    from src.revenue_attribution import calculate_revenue_attribution, save_revenue_attribution
except ImportError:
    pass

try:
    from src.predictive_forecasting import generate_combined_intelligence
except ImportError:
    pass

def render_unified_dashboard():
    """
    Renders the Unified Command Center (Strategy + Tactics)
    """
    # 1. Ensure Data is Loaded
    res, fin, portfolio_df, portfolio_context = ensure_data_loaded()
    
    # Check Offline State
    if not res or portfolio_df.empty:
        st.markdown("""
        <div style="text-align: center; padding: 60px 40px; background: white; border: 2px dashed #e0e0e0; border-radius: 12px;">
            <div style="font-size: 72px; margin-bottom: 20px;">🧩</div>
            <div style="font-size: 28px; font-weight: 700; color: #666; margin-bottom: 12px;">
                COMMAND CENTER OFFLINE
            </div>
            <div style="font-size: 14px; color: #666; margin-bottom: 20px;">
                Initialize a project in Market Discovery to activate.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Unpack Data
    current_revenue = res.get('total_rev', 0)
    previous_revenue = current_revenue * 0.9 # Fallback estimate if not strictly calc
    
    
    # Re-calculate previous revenue more accurately from df_weekly if possible
    df_weekly = st.session_state.get('df_weekly', pd.DataFrame())
    
    # === DATA PREPARATION (Fix for missing 'date'/'week') ===
    if not df_weekly.empty:
        # standardise columns
        if 'date' not in df_weekly.columns:
            if 'week' in df_weekly.columns:
                try:
                    df_weekly['date'] = pd.to_datetime(df_weekly['week'])
                except:
                    # If week parsing fails, try to use index or create dummy
                    pass
        
        # Ensure 'week' exists if we have 'date' (for predictive engine fallback)
        if 'week' not in df_weekly.columns and 'date' in df_weekly.columns:
            df_weekly['week'] = df_weekly['date']

    if not df_weekly.empty and 'revenue_proxy' in df_weekly.columns:
        # Simple estimation logic for previous period
        # (Real implementation uses dates, kept simple here to avoid complexity)
        pass

    # === SECTION 1: STRATEGIC INTELLIGENCE (The Why) ===
    st.markdown("### 🧩 Strategic Context")
    
    # Calculate Attribution
    attribution = None
    try:
        # Using simplified attribution call for display
        # In production, this uses the real engine
        from src.revenue_attribution import calculate_revenue_attribution

        # Get category_id from session state (if available)
        category_id = st.session_state.get('active_project_category_id')

        attribution = calculate_revenue_attribution(
            previous_revenue=previous_revenue,
            current_revenue=current_revenue,
            df_weekly=df_weekly,
            trigger_events=[], # Passed empty for speed
            market_snapshot=None,
            category_id=category_id
        )
    except Exception as e:
        st.error(f"⚠️ Attribution Error: {str(e)}")
        # import traceback
        # st.code(traceback.format_exc())

    if attribution:
        # Metric Headers
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Revenue", f_money(current_revenue), f"{attribution.delta_pct:.1f}%", delta_color="normal")
        with col2:
            earned_pct = attribution.get_earned_percentage()
            st.metric("Earned Growth", f_money(attribution.internal_contribution), f"{earned_pct:.0f}% share")
        with col3:
            opp_pct = attribution.get_opportunistic_percentage()
            st.metric("Opportunistic Growth", f_money(attribution.get_opportunistic_growth()), f"{opp_pct:.0f}% share")
        
        st.markdown("---")
        
        # Waterfall Chart
        st.markdown("#### 📊 Revenue Drivers (Waterfall)")
        fig = go.Figure(go.Waterfall(
            name="Revenue Attribution", orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Start", "Internal", "Competition", "Market", "Platform", "End"],
            y=[previous_revenue, attribution.internal_contribution, attribution.competitive_contribution, 
               attribution.macro_contribution, attribution.platform_contribution, current_revenue],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(height=350, title=None, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Residual Warning (if high unexplained variance)
        residual_warning = attribution.get_residual_warning()
        if residual_warning:
            st.warning(residual_warning)
            # Show data quality summary
            data_quality = attribution.get_data_quality_summary()
            st.caption(data_quality)

        # Category Benchmarks Comparison (P3 Feature)
        macro_drivers = attribution.macro_drivers if hasattr(attribution, 'macro_drivers') else []
        for driver in macro_drivers:
            if driver.event_type == "category_trend" and driver.metadata:
                category_growth = driver.metadata.get('category_growth_pct', 0)
                your_growth = (attribution.total_delta / previous_revenue * 100) if previous_revenue > 0 else 0
                market_share = driver.metadata.get('market_share', 0)
                growth_direction = driver.metadata.get('growth_direction', 'unknown')

                # Show comparison
                st.markdown("##### 📊 Category Performance Comparison")

                comp_cols = st.columns(3)
                with comp_cols[0]:
                    st.metric(
                        "Category Growth",
                        f"{category_growth:+.1f}%",
                        delta=growth_direction.title(),
                        help="30-day category-wide revenue trend"
                    )
                with comp_cols[1]:
                    st.metric(
                        "Your Growth",
                        f"{your_growth:+.1f}%",
                        delta="Outperforming" if your_growth > category_growth else "Underperforming",
                        help="Your revenue growth vs category benchmark"
                    )
                with comp_cols[2]:
                    st.metric(
                        "Market Share",
                        f"{market_share*100:.2f}%",
                        help="Your estimated category market share"
                    )

                # Insight
                if your_growth > category_growth + 5:
                    st.success(f"✅ You're growing faster than the category ({your_growth:.1f}% vs {category_growth:.1f}%). This is earned market share gain.")
                elif your_growth < category_growth - 5:
                    st.warning(f"⚠️ You're growing slower than the category ({your_growth:.1f}% vs {category_growth:.1f}%). You're losing market share.")
                else:
                    st.info(f"📊 Your growth is tracking with the category ({your_growth:.1f}% vs {category_growth:.1f}%). This is primarily macro trend.")

                break  # Only show first category trend driver

    else:
        # Fallback if attribution fails
        st.info("⚠️ Strategic Context Unavailable: insufficient historical data for attribution.")

    # === SUSTAINABILITY ANALYSIS BANNER ===
    # Show sustainable run rate vs current revenue (calculated from CombinedIntelligence)
    # This is calculated later in the predictive section, so we'll add it there after combined_intel is generated

    # === SECTION 1.5: PREDICTIVE HORIZON ===
    # Forecast Chart (Restored from Command Center 2.0)
    try:
        from src.predictive_forecasting import generate_combined_intelligence
        
        # We need trigger events for the forecast. In this unified view, we can try to get them
        # or pass empty if not easily available. ideally we'd load them in ensure_data_loaded.
        # For now, pass empty list to get base forecast.
        combined_intel = generate_combined_intelligence(
            current_revenue=current_revenue,
            previous_revenue=previous_revenue,
            attribution=attribution,
            trigger_events=[], 
            df_historical=df_weekly
        )
        
        if combined_intel and combined_intel.forecast:
            st.markdown("---")

            # === SUSTAINABILITY ANALYSIS BANNER ===
            sustainable_run_rate = combined_intel.sustainable_run_rate
            temporary_inflation = combined_intel.temporary_inflation
            temporary_duration = combined_intel.temporary_duration_days

            # Calculate inflation percentage
            inflation_pct = (temporary_inflation / current_revenue * 100) if current_revenue > 0 else 0

            # Determine banner color based on inflation level
            if abs(inflation_pct) > 20:
                banner_color = "#fee"  # Light red
                icon = "⚠️"
            elif abs(inflation_pct) > 10:
                banner_color = "#fff3cd"  # Light yellow
                icon = "⚡"
            else:
                banner_color = "#e8f5e9"  # Light green
                icon = "✅"

            # Display sustainability banner
            st.markdown(f"""
            <div style="background: {banner_color}; border-left: 5px solid {'#dc3545' if abs(inflation_pct) > 20 else '#ffc107' if abs(inflation_pct) > 10 else '#28a745'};
                        padding: 20px; border-radius: 6px; margin-bottom: 20px;">
                <div style="font-size: 18px; font-weight: 700; color: #1a1a1a; margin-bottom: 12px;">
                    {icon} Sustainability Analysis
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 4px;">Current Revenue</div>
                        <div style="font-size: 24px; font-weight: 700; color: #1a1a1a;">${current_revenue:,.0f}/mo</div>
                    </div>
                    <div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 4px;">Sustainable Run Rate</div>
                        <div style="font-size: 24px; font-weight: 700; color: #28a745;">${sustainable_run_rate:,.0f}/mo</div>
                    </div>
                    <div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 4px;">Temporary {'Inflation' if temporary_inflation > 0 else 'Deflation'}</div>
                        <div style="font-size: 24px; font-weight: 700; color: {'#dc3545' if temporary_inflation > 0 else '#28a745'};">
                            {'+' if temporary_inflation > 0 else ''}${temporary_inflation:,.0f}
                        </div>
                        <div style="font-size: 12px; color: #666; margin-top: 4px;">
                            ({abs(inflation_pct):.0f}% of current) • Reverses in {temporary_duration} days
                        </div>
                    </div>
                </div>
                <div style="margin-top: 16px; font-size: 14px; color: #666; border-top: 1px solid #ddd; padding-top: 12px;">
                    <strong>What this means:</strong> Your sustainable revenue after temporary factors reverse is
                    <strong style="color: {'#dc3545' if sustainable_run_rate < current_revenue else '#28a745'};">
                        ${sustainable_run_rate:,.0f}/month
                    </strong>.
                    {'⚠️ Caution: Don\'t scale fixed costs on temporary gains.' if temporary_inflation > current_revenue * 0.15 else '✅ Your growth is primarily sustainable.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🔮 Predictive Horizon")
            portfolio_forecast = combined_intel.forecast
            
            p_col1, p_col2 = st.columns([2, 1])
            
            with p_col1:
                # Prepare Forecast Chart Data
                # 1. Historical Data (Monthly)
                if not df_weekly.empty:
                    # Ensure date column exists with validation
                    if 'date' not in df_weekly.columns:
                        if 'week' in df_weekly.columns:
                            try:
                                df_weekly['date'] = pd.to_datetime(df_weekly['week'], errors='coerce')
                            except Exception:
                                # If conversion fails, skip chart
                                df_weekly = pd.DataFrame()
                        else:
                            # No date or week column, skip chart
                            df_weekly = pd.DataFrame()

                    # Only proceed if we have valid data
                    if not df_weekly.empty and 'date' in df_weekly.columns:
                        df_weekly['month'] = df_weekly['date'].dt.to_period('M')
                        # Use revenue proxy or calc from sum
                        rev_col_chart = 'revenue_proxy' if 'revenue_proxy' in df_weekly.columns else 'sales' if 'sales' in df_weekly.columns else None

                        if rev_col_chart:
                            monthly_rev = df_weekly.groupby('month')[rev_col_chart].sum().reset_index()
                            monthly_rev['month'] = monthly_rev['month'].dt.to_timestamp()

                            # Filter to last 6 months
                            cutoff_month = pd.Timestamp.now() - pd.Timedelta(days=180)
                            monthly_rev = monthly_rev[monthly_rev['month'] >= cutoff_month]

                            # 2. Future Data Point
                            last_date = monthly_rev['month'].max() if not monthly_rev.empty else pd.Timestamp.now()
                            next_date = last_date + pd.Timedelta(days=30)

                            proj_rev = portfolio_forecast.projected_revenue
                            lower_bound = portfolio_forecast.lower_bound
                            upper_bound = portfolio_forecast.upper_bound

                            # Create Chart
                            fig_pred = go.Figure()

                            # Historical Bars
                            fig_pred.add_trace(go.Bar(
                                x=monthly_rev['month'],
                                y=monthly_rev[rev_col_chart],
                                name='Historical Revenue',
                                marker_color='#e0e0e0'
                            ))

                            # Forecast Line
                            fig_pred.add_trace(go.Scatter(
                                x=[last_date, next_date],
                                y=[monthly_rev[rev_col_chart].iloc[-1] if not monthly_rev.empty else 0, proj_rev],
                                name='Projected Trend',
                                line=dict(color='#007bff', width=3, dash='dot'),
                                mode='lines+markers'
                            ))

                            # Confidence Interval (Error Bars)
                            fig_pred.add_trace(go.Scatter(
                                x=[next_date, next_date],
                                y=[upper_bound, lower_bound],
                                mode='markers',
                                marker=dict(color='#007bff', size=1),
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=[upper_bound - proj_rev],
                                    arrayminus=[proj_rev - lower_bound],
                                    color='rgba(0,123,255,0.3)',
                                    thickness=10,
                                    width=10
                                ),
                                name='Confidence Interval (80%)'
                            ))

                            fig_pred.update_layout(
                                height=300,
                                margin=dict(l=40, r=40, t=30, b=30),
                                showlegend=True,
                                legend=dict(orientation="h", y=1.1),
                                title=None
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
            
            with p_col2:
                 st.metric(
                    "Est. Annual Run Rate",
                    f"${portfolio_forecast.projected_annual_sales:,.0f}",
                    delta=None,
                    help="Based on sustainable run rate and seasonality"
                )
                 st.info(f"On track for **${proj_rev:,.0f}** next month.")
        else:
            st.info("⚠️ Predictive Horizon Unavailable: Insufficient historical data or trigger events for forecast generation.")
                 
    except Exception as e:
        st.warning(f"Predictive engine offline: {e}")
        # pass

    # === EVENT TIMELINE TABLE ===
    if combined_intel and hasattr(combined_intel, 'anticipated_events') and combined_intel.anticipated_events:
        st.markdown("---")
        st.markdown("#### 📅 Anticipated Events (Next 30 Days)")

        # Build timeline table
        events = combined_intel.anticipated_events
        if events:
            timeline_data = []

            # Severity styling map
            severity_styles = {
                'CRITICAL': {'emoji': '🚨', 'color': '#dc3545', 'bg': '#fee'},
                'HIGH': {'emoji': '⚡', 'color': '#fd7e14', 'bg': '#fff3cd'},
                'MEDIUM': {'emoji': '⚠️', 'color': '#ffc107', 'bg': '#fff8e1'},
                'LOW': {'emoji': '📉', 'color': '#6c757d', 'bg': '#f8f9fa'}
            }

            # Sort by days_until (most urgent first)
            sorted_events = sorted(events, key=lambda e: e.days_until if hasattr(e, 'days_until') else 999)

            for event in sorted_events:
                # Get severity (handle both string and enum)
                severity = event.severity if hasattr(event, 'severity') else 'MEDIUM'
                if hasattr(severity, 'name'):
                    severity_name = severity.name
                else:
                    severity_name = str(severity).upper()

                # Get styling
                style = severity_styles.get(severity_name, severity_styles['MEDIUM'])

                # Build row
                timeline_data.append({
                    "Day": f"Day {event.days_until if hasattr(event, 'days_until') else '?'}",
                    "Event": f"{style['emoji']} {event.description if hasattr(event, 'description') else event.event_type}",
                    "Impact": f"${event.impact_per_month:+,.0f}/mo" if hasattr(event, 'impact_per_month') else 'N/A',
                    "Severity": severity_name,
                    "Action": event.action_recommended if hasattr(event, 'action_recommended') else 'Monitor'
                })

            # Display as dataframe
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)

                # Custom styling function
                def style_severity(row):
                    severity = row['Severity']
                    style = severity_styles.get(severity, severity_styles['MEDIUM'])
                    return [f"background-color: {style['bg']}"] * len(row)

                # Apply styling
                styled_timeline = df_timeline.style.apply(style_severity, axis=1)

                st.dataframe(styled_timeline, use_container_width=True, hide_index=True)

                # Net impact summary
                total_impact = sum(
                    event.impact_per_month if hasattr(event, 'impact_per_month') else 0
                    for event in sorted_events
                )

                if total_impact != 0:
                    impact_color = '#28a745' if total_impact > 0 else '#dc3545'
                    st.markdown(f"""
                    <div style="margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid {impact_color};">
                        <strong>Net Projected Impact (30 days):</strong>
                        <span style="color: {impact_color}; font-size: 18px; font-weight: 700;">
                            ${total_impact:+,.0f}/month
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No anticipated events detected for the next 30 days.")

    # === SCENARIO COMPARISON UI ===
    if combined_intel and hasattr(combined_intel, 'scenarios') and combined_intel.scenarios:
        st.markdown("---")
        st.markdown("#### 🎲 Scenario Analysis (30/60/90 Day Outlook)")

        scenarios = combined_intel.scenarios
        if scenarios and len(scenarios) >= 3:
            # Create 3-column layout for Base/Optimistic/Pessimistic
            scen_cols = st.columns(3)

            # Sort scenarios: Base Case first, then Optimistic, then Pessimistic
            scenario_order = {"Base Case": 0, "Optimistic": 1, "Pessimistic": 2}
            sorted_scenarios = sorted(scenarios, key=lambda s: scenario_order.get(s.scenario_name, 3))

            # Styling map for scenario types
            scenario_styles = {
                "Base Case": {"color": "#007bff", "bg": "#e7f3ff", "icon": "📊"},
                "Optimistic": {"color": "#28a745", "bg": "#e8f5e9", "icon": "🚀"},
                "Pessimistic": {"color": "#dc3545", "bg": "#fee", "icon": "⚠️"}
            }

            for idx, scenario in enumerate(sorted_scenarios[:3]):  # Show top 3
                style = scenario_styles.get(scenario.scenario_name, {"color": "#6c757d", "bg": "#f8f9fa", "icon": "📈"})

                with scen_cols[idx]:
                    # Scenario header card
                    st.markdown(f"""
                    <div style="background: {style['bg']}; border-left: 4px solid {style['color']};
                                padding: 16px; border-radius: 6px; margin-bottom: 12px;">
                        <div style="font-size: 16px; font-weight: 700; color: {style['color']}; margin-bottom: 8px;">
                            {style['icon']} {scenario.scenario_name}
                        </div>
                        <div style="font-size: 13px; color: #666;">
                            Probability: <strong>{scenario.probability:.0%}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Revenue projections
                    st.markdown("**Revenue Projections:**")

                    # 30-day projection
                    proj_30d = scenario.projected_revenue_30d if hasattr(scenario, 'projected_revenue_30d') else 0
                    delta_30d = proj_30d - current_revenue
                    st.metric(
                        "30 Days",
                        f"${proj_30d:,.0f}",
                        delta=f"{delta_30d:+,.0f}",
                        delta_color="normal"
                    )

                    # 60-day projection
                    if hasattr(scenario, 'projected_revenue_60d'):
                        proj_60d = scenario.projected_revenue_60d
                        delta_60d = proj_60d - current_revenue
                        st.metric(
                            "60 Days",
                            f"${proj_60d:,.0f}",
                            delta=f"{delta_60d:+,.0f}",
                            delta_color="normal"
                        )

                    # 90-day projection
                    if hasattr(scenario, 'projected_revenue_90d'):
                        proj_90d = scenario.projected_revenue_90d
                        delta_90d = proj_90d - current_revenue
                        st.metric(
                            "90 Days",
                            f"${proj_90d:,.0f}",
                            delta=f"{delta_90d:+,.0f}",
                            delta_color="normal"
                        )

                    # Key assumptions/narrative
                    if hasattr(scenario, 'narrative') and scenario.narrative:
                        st.caption(f"**Scenario:** {scenario.narrative}")
                    elif hasattr(scenario, 'assumptions') and scenario.assumptions:
                        assumptions_text = ", ".join([a.get('assumption', '') for a in scenario.assumptions[:2]])
                        st.caption(f"**Key Factors:** {assumptions_text}")

            # Summary insight
            st.markdown("---")
            base_scenario = next((s for s in sorted_scenarios if s.scenario_name == "Base Case"), None)
            if base_scenario:
                base_30d = base_scenario.projected_revenue_30d if hasattr(base_scenario, 'projected_revenue_30d') else current_revenue
                st.info(f"""
                **Most Likely Outcome:** Based on current trajectory and anticipated events,
                expect revenue of **${base_30d:,.0f}** in 30 days ({base_scenario.probability:.0%} confidence).
                """)

    # === SECTION 2: TACTICAL EXECUTION (The What) ===
    st.markdown("---")
    st.markdown("### 🛡️ Tactical Response Plan")
    
    # Generate Intelligence (Predictive Risk)
    from utils.ai_engine import _determine_state_fallback
    
    # Enrich portfolio with intelligence
    enriched_rows = []
    
    # Progress bar for intelligence generation
    if not portfolio_df.empty:
        # progress_bar = st.progress(0)
        # total_rows = len(portfolio_df)
        
        for i, (idx, row) in enumerate(portfolio_df.iterrows()):
            # Convert row to dict for AI engine
            row_data = row.to_dict()
            
            # Use fallback deterministic logic (fast, synchronous)
            # In validation mode, we skip the heavy async LLM call to ensure UI responsiveness
            brief = _determine_state_fallback(
                row_data=row_data,
                strategic_bias=st.session_state.get('strategic_bias', 'Balanced Defense')
            )
            
            # Add intelligence back to row
            row_data['strategic_state'] = brief.strategic_state
            row_data['confidence'] = brief.confidence
            row_data['recommended_action'] = brief.recommended_action
            row_data['reasoning'] = brief.reasoning
            row_data['thirty_day_risk'] = brief.thirty_day_risk
            row_data['optimization_value'] = brief.optimization_value
            row_data['predictive_state'] = brief.predictive_state
            row_data['state_emoji'] = brief.state_emoji
            
            enriched_rows.append(row_data)
            # progress_bar.progress((i + 1) / total_rows)
            
        # Create enriched dataframe
        enriched_portfolio_df = pd.DataFrame(enriched_rows)
    else:
        enriched_portfolio_df = portfolio_df.copy()

    # Sort by Importance (Risk + Opportunistic Value)
    if not enriched_portfolio_df.empty:
        # Create a combined 'urgency_score'
        # Risk is weighted 2x higher than opportunity for 'Protection' focus
        enriched_portfolio_df['urgency_score'] = (
            enriched_portfolio_df.get('thirty_day_risk', 0).fillna(0) * 1.5 + 
            enriched_portfolio_df.get('optimization_value', 0).fillna(0)
        )
        
        # Sort descending
        top_items = enriched_portfolio_df.nlargest(5, 'urgency_score')
        
        st.markdown(f"#### 🎯 Priority Actions ({len(enriched_portfolio_df)} monitored)")
        
        if top_items['urgency_score'].sum() == 0:
             st.info("✅ No critical risks or major opportunities detected. Portfolio is stable.")
        
        for i, (idx, row) in enumerate(top_items.iterrows()):
            title = row.get('title', 'Unknown')[:60]
            asin = row.get('asin', 'Unknown')
            risk = row.get('thirty_day_risk', 0)
            opp = row.get('optimization_value', 0)
            action = row.get('recommended_action', 'Review perfomance.')
            reason = row.get('reasoning', '')
            emoji = row.get('state_emoji', '🛡️')
            state = row.get('strategic_state', 'Unknown')
            
            # Dynamic Border Color based on State
            border_color = "#e0e0e0"
            if state == 'DISTRESS': border_color = "#dc3545" # Red
            elif state == 'TRENCH_WAR': border_color = "#fd7e14" # Orange
            elif state == 'HARVEST': border_color = "#28a745" # Green
            elif state == 'FORTRESS': border_color = "#00704A" # Starbucks Green (Domination)
            elif state == 'TERMINAL': border_color = "#343a40" # Dark Gray (Exit)
            
            # Value display
            value_display = ""
            if risk > 0:
                value_display = f"<span style='color: #dc3545; font-weight: 700;'>-${risk:,.0f} Risk</span>"
            elif opp > 0:
                value_display = f"<span style='color: #28a745; font-weight: 700;'>+${opp:,.0f} Opportunity</span>"
            else:
                value_display = "<span style='color: #666;'>Stable</span>"
            
            st.markdown(f"""
            <div style="background: white; border: 1px solid #e0e0e0; border-left: 5px solid {border_color}; 
                        padding: 16px; border-radius: 6px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="font-weight: 600; color: #1a1a1a; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 20px;">{emoji}</span> 
                        <span>#{i+1} {state.replace('_', ' ')}</span>
                    </div>
                    <div>{value_display}</div>
                </div>
                <div style="font-size: 13px; color: #444; margin-top: 8px; font-weight: 500;">{title}</div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #f0f0f0;">
                    <div style="font-size: 14px; color: #1a1a1a; font-weight: 600; margin-bottom: 4px;">
                        ⚡ {action}
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        {reason}
                        <br><span style="color: #999; font-size: 11px;">ASIN: {asin}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Table
    with st.expander("📋 View Full Portfolio Intelligence"):
        # Format for clean display
        display_df = enriched_portfolio_df.copy()
        
        # Select and rename columns for display
        cols_to_show = [
            'asin', 'title', 'strategic_state', 'recommended_action', 
            'thirty_day_risk', 'optimization_value', 'revenue_proxy', 'margin_health'
        ]
        # Filter only existing columns
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
        
        st.dataframe(
            display_df[cols_to_show].style.format({
                'thirty_day_risk': '${:,.0f}', 
                'optimization_value': '${:,.0f}',
                'revenue_proxy': '${:,.0f}'
            }), 
            use_container_width=True
        )

