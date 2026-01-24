"""
Unified Intelligence Pipeline Orchestrator

Master pipeline that orchestrates all intelligence systems:
1. Trigger Detection → Identify market changes
2. Network Intelligence → Get category benchmarks
3. Strategic Triangulator → Unified AI Classification + Insight Generation
4. Revenue Attribution → Causal decomposition ("Why it happened")
5. Predictive Forecasting → Forward-looking scenarios ("What's next")
6. Database Storage → Store unified intelligence

REFACTORED: Now uses the main StrategicTriangulator instead of separate v2/v3 engines.
Enhanced with Causal Intelligence Platform (Phase 2-2.5) integration.

This is the main entry point for generating portfolio intelligence.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from supabase import Client

from src.models.product_status import ProductStatus
from src.models.trigger_event import TriggerEvent
from src.models.unified_intelligence import UnifiedIntelligence
from src.trigger_detection import detect_trigger_events
from src.network_intelligence import NetworkIntelligence
from src.data_accumulation import NetworkIntelligenceAccumulator

# REFACTORED: Use unified StrategicTriangulator instead of separate v2/v3 engines
from utils.ai_engine import StrategicTriangulator, StrategicState

# PHASE 2-2.5: Causal Intelligence and Predictive Forecasting
try:
    from src.revenue_attribution import calculate_revenue_attribution
    from src.predictive_forecasting import (
        generate_combined_intelligence,
        forecast_event_impacts,
        build_scenarios,
        calculate_sustainable_run_rate
    )
    CAUSAL_INTELLIGENCE_ENABLED = True
except ImportError:
    CAUSAL_INTELLIGENCE_ENABLED = False
    calculate_revenue_attribution = None
    generate_combined_intelligence = None


class IntelligencePipeline:
    """
    Unified intelligence pipeline orchestrator.

    Usage:
        pipeline = IntelligencePipeline(supabase_client, openai_api_key)
        results = pipeline.generate_portfolio_intelligence(portfolio_asins, market_data)
    """

    def __init__(
        self,
        supabase: Client,
        openai_api_key: Optional[str] = None,
        enable_data_accumulation: bool = True
    ):
        """
        Initialize intelligence pipeline.

        Args:
            supabase: Supabase client for database operations
            openai_api_key: OpenAI API key for LLM calls
            enable_data_accumulation: Whether to accumulate network intelligence
        """
        self.supabase = supabase
        self.enable_data_accumulation = enable_data_accumulation

        # Initialize subsystems
        self.network_intelligence = NetworkIntelligence(supabase)
        self.accumulator = NetworkIntelligenceAccumulator(supabase) if enable_data_accumulation else None
        
        # REFACTORED: Use unified StrategicTriangulator with triggers and network enabled
        self.triangulator = StrategicTriangulator(
            use_llm=True,
            strategic_bias="Balanced Defense",
            enable_triggers=True,
            enable_network=True
        )

    def generate_portfolio_intelligence(
        self,
        portfolio_asins: List[str],
        market_data: Dict[str, Any],
        category_context: Dict[str, Any]
    ) -> List[UnifiedIntelligence]:
        """
        Generate unified intelligence for entire portfolio.

        Args:
            portfolio_asins: List of ASINs to analyze
            market_data: Dict mapping ASIN → historical data, competitor data
            category_context: Category metadata (category_id, name, tree)

        Returns:
            List of UnifiedIntelligence objects (one per ASIN)
        """

        print(f"\n🧠 Intelligence Pipeline: Analyzing {len(portfolio_asins)} ASINs...")

        results = []

        for i, asin in enumerate(portfolio_asins, 1):
            print(f"\n[{i}/{len(portfolio_asins)}] Processing {asin}...")

            try:
                # Generate intelligence for single ASIN
                intelligence = self.generate_single_asin_intelligence(
                    asin=asin,
                    historical_data=market_data.get(asin, {}).get('historical', pd.DataFrame()),
                    competitor_data=market_data.get(asin, {}).get('competitors', pd.DataFrame()),
                    current_metrics=market_data.get(asin, {}).get('current_metrics', {}),
                    category_context=category_context
                )

                if intelligence:
                    results.append(intelligence)

                    # Store in database
                    self._store_intelligence(intelligence)

            except Exception as e:
                print(f"⚠️  Error processing {asin}: {str(e)}")
                continue

        print(f"\n✅ Pipeline complete: {len(results)}/{len(portfolio_asins)} ASINs processed")

        return results

    def generate_single_asin_intelligence(
        self,
        asin: str,
        historical_data: pd.DataFrame,
        competitor_data: pd.DataFrame,
        current_metrics: Dict[str, Any],
        category_context: Dict[str, Any]
    ) -> Optional[UnifiedIntelligence]:
        """
        Generate unified intelligence for a single ASIN.

        REFACTORED: Now uses unified StrategicTriangulator which combines:
        - Trigger detection
        - Network intelligence
        - Strategic classification
        - Insight generation
        
        All in a single call, eliminating separate v2/v3 engines.

        Args:
            asin: Product ASIN
            historical_data: 90-day historical DataFrame
            competitor_data: Current competitor DataFrame
            current_metrics: Current snapshot dict
            category_context: Category metadata

        Returns:
            UnifiedIntelligence object or None if failed
        """

        try:
            # STEP 1: Get network intelligence (for benchmarks)
            print(f"  🌐 Querying network intelligence...")
            category_id = category_context.get('category_id')
            network_intel = self._gather_network_intelligence(asin, category_id, current_metrics)

            # STEP 2: Prepare row data for StrategicTriangulator
            # The triangulator expects a dict with all metrics + historical/competitor data
            row_data = {
                'asin': asin,
                **current_metrics,
                # Inject historical and competitor data for trigger detection
                'historical_df': historical_data,
                'competitors_df': competitor_data,
                # Inject network intelligence for competitive context
                'category_id': category_id,
                **network_intel.get('competitive_position', {}),
            }
            
            # Add category benchmarks to row data
            benchmarks = network_intel.get('category_benchmarks', {})
            if benchmarks:
                row_data['median_price'] = benchmarks.get('median_price')
                row_data['median_bsr'] = benchmarks.get('median_bsr')
                row_data['median_review_count'] = benchmarks.get('median_review_count')

            # STEP 3: Run unified StrategicTriangulator
            print(f"  🤖 Running unified AI analysis...")
            revenue = current_metrics.get('estimated_monthly_revenue', 0)
            brief = self.triangulator.analyze(row_data, revenue=revenue)

            print(f"     State: {brief.strategic_state}")
            print(f"     Confidence: {brief.confidence:.0f}%")
            print(f"     30-Day Risk: ${brief.thirty_day_risk:.2f}")
            print(f"     30-Day Growth: ${brief.thirty_day_growth:.2f}")

            # STEP 4: Detect trigger events (for storage/display)
            print(f"  🔍 Detecting trigger events...")
            trigger_events = detect_trigger_events(
                asin=asin,
                df_historical=historical_data,
                df_competitors=competitor_data,
                lookback_days=30
            )
            print(f"     Found {len(trigger_events)} trigger events")

            # STEP 4.5: CAUSAL ATTRIBUTION (Why it happened)
            attribution = None
            if CAUSAL_INTELLIGENCE_ENABLED and calculate_revenue_attribution:
                print(f"  📊 Calculating revenue attribution...")
                try:
                    previous_revenue = current_metrics.get('previous_monthly_revenue', revenue * 0.9)
                    attribution = calculate_revenue_attribution(
                        previous_revenue=previous_revenue,
                        current_revenue=revenue,
                        df_weekly=historical_data,
                        trigger_events=trigger_events,
                        market_snapshot=competitor_data.to_dict('records') if not competitor_data.empty else None,
                        lookback_days=30
                    )
                    if attribution:
                        print(f"     Internal: ${attribution.internal_contribution:,.0f} | Competitive: ${attribution.competitive_contribution:,.0f}")
                        print(f"     Macro: ${attribution.macro_contribution:,.0f} | Platform: ${attribution.platform_contribution:,.0f}")
                except Exception as e:
                    print(f"     ⚠️ Attribution failed: {str(e)}")

            # STEP 4.6: PREDICTIVE FORECASTING (What's next)
            combined_intel = None
            anticipated_events = []
            scenarios = []
            sustainable_run_rate = revenue
            
            if CAUSAL_INTELLIGENCE_ENABLED and generate_combined_intelligence:
                print(f"  🔮 Generating predictive forecast...")
                try:
                    combined_intel = generate_combined_intelligence(
                        current_revenue=revenue,
                        previous_revenue=previous_revenue if 'previous_revenue' in dir() else revenue * 0.9,
                        attribution=attribution,
                        trigger_events=trigger_events,
                        df_historical=historical_data
                    )
                    if combined_intel:
                        anticipated_events = combined_intel.anticipated_events
                        scenarios = combined_intel.scenarios
                        sustainable_run_rate = combined_intel.sustainable_run_rate
                        print(f"     Sustainable Run Rate: ${sustainable_run_rate:,.0f}/mo")
                        print(f"     Anticipated Events: {len(anticipated_events)}")
                        print(f"     Scenarios: {len(scenarios)}")
                except Exception as e:
                    print(f"     ⚠️ Forecasting failed: {str(e)}")

            # STEP 5: Map strategic state to ProductStatus
            state_to_status = {
                "FORTRESS": ProductStatus.STABLE,
                "HARVEST": ProductStatus.OPPORTUNITY,
                "TRENCH_WAR": ProductStatus.WATCH,
                "DISTRESS": ProductStatus.CRITICAL,
                "TERMINAL": ProductStatus.CRITICAL,
            }
            product_status = state_to_status.get(brief.strategic_state, ProductStatus.WATCH)

            # STEP 6: Calculate net expected value (enhanced with sustainable run rate)
            net_ev = brief.thirty_day_growth - brief.thirty_day_risk
            if sustainable_run_rate != revenue:
                # Adjust net EV based on sustainable run rate
                temporary_component = revenue - sustainable_run_rate
                net_ev -= temporary_component * 0.5  # Discount temporary gains
            print(f"     Net EV: ${net_ev:+.2f}/mo")

            # STEP 7: Determine action type from strategic state
            state_to_action = {
                "FORTRESS": "defend",
                "HARVEST": "optimize",
                "TRENCH_WAR": "repair",
                "DISTRESS": "repair",
                "TERMINAL": "exit",
            }
            action_type = state_to_action.get(brief.strategic_state, "monitor")

            # STEP 8: Combine into UnifiedIntelligence (with causal + predictive data)
            unified = UnifiedIntelligence(
                # Identity
                asin=asin,
                timestamp=datetime.now(),

                # Trigger events
                trigger_events=trigger_events,
                primary_trigger=trigger_events[0].event_type if trigger_events else None,

                # Strategic classification (from StrategicTriangulator)
                product_status=product_status,
                strategic_state=brief.strategic_state,
                confidence=brief.confidence,
                reasoning=brief.reasoning,

                # Predictive intelligence (from StrategicTriangulator + Forecasting)
                thirty_day_risk=brief.thirty_day_risk,
                thirty_day_growth=brief.thirty_day_growth,
                net_expected_value=net_ev,

                # Actionable insight (from StrategicTriangulator)
                recommendation=brief.ai_recommendation or brief.recommended_plan,
                projected_upside_monthly=brief.thirty_day_growth,
                downside_risk_monthly=brief.thirty_day_risk,
                action_type=action_type,
                time_horizon_days=30 if brief.strategic_state in ["DISTRESS", "TERMINAL"] else 90,

                # Network context (metadata only)
                category_benchmarks_summary=self._summarize_category_benchmarks(
                    network_intel.get('category_benchmarks', {})
                ),
                competitive_position_summary=self._summarize_competitive_position(
                    network_intel.get('competitive_position', {})
                ),
                
                # NEW: Causal Attribution (Phase 2)
                revenue_attribution=attribution,
                
                # NEW: Predictive Forecasting (Phase 2.5)
                anticipated_events=anticipated_events,
                scenarios=scenarios,
                sustainable_run_rate=sustainable_run_rate,
                combined_intelligence=combined_intel
            )

            # Persist to database
            if self.supabase:
                print(f"  💾 Persisting intelligence to Supabase...")
                self.persist_intelligence(unified)

            return unified

        except Exception as e:
            print(f"⚠️  Error generating intelligence for {asin}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def accumulate_market_data(
        self,
        market_snapshot: pd.DataFrame,
        category_id: int,
        category_name: str,
        category_tree: List[str]
    ) -> None:
        """
        Accumulate market data into network intelligence.

        Call this after Phase 2 discovery to store data for future use.

        Args:
            market_snapshot: DataFrame of all products discovered
            category_id: Keepa category ID
            category_name: Category name
            category_tree: Category hierarchy
        """

        if not self.enable_data_accumulation or not self.accumulator:
            print("⚠️  Data accumulation disabled")
            return

        print(f"\n📊 Accumulating market data for category {category_id}...")

        try:
            self.accumulator.accumulate_search_data(
                market_snapshot=market_snapshot,
                category_id=category_id,
                category_name=category_name,
                category_tree=category_tree
            )
            print(f"✅ Data accumulated successfully")

        except Exception as e:
            print(f"⚠️  Error accumulating data: {str(e)}")

    def persist_intelligence(self, intelligence: UnifiedIntelligence) -> bool:
        """
        Persist unified intelligence to Supabase.
        
        Writes to:
        1. strategic_insights (AI analysis, attribution, forecasting)
        2. trigger_events (detected market events)
        
        Args:
            intelligence: UnifiedIntelligence object to save
            
        Returns:
            bool: True if successful
        """
        if not self.supabase:
            return False
            
        try:
            # 1. Save Strategic Insight
            insight_data = intelligence.to_database_record()
            
            # Add expiration (default 30 days)
            insight_data['expires_at'] = (datetime.now() + timedelta(days=30)).isoformat()
            
            result = self.supabase.table('strategic_insights').insert(insight_data).execute()
            
            if not result.data:
                print(f"⚠️ Failed to persist insight for {intelligence.asin}")
                return False
                
            insight_id = result.data[0]['id']
            print(f"✅ Persisted insight {insight_id} for {intelligence.asin}")
            
            # 2. Save Trigger Events (if any)
            if intelligence.trigger_events:
                events_data = []
                for event in intelligence.trigger_events:
                    event_dict = event.to_dict()
                    # Add foreign key to parent insight
                    event_dict['generated_insight_id'] = insight_id
                    # Ensure ASIN is set
                    event_dict['asin'] = intelligence.asin
                    events_data.append(event_dict)
                    
                if events_data:
                    self.supabase.table('trigger_events').insert(events_data).execute()
                    print(f"   Saved {len(events_data)} trigger events")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error persisting intelligence for {intelligence.asin}: {str(e)}")
            return False

    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================

    def _gather_network_intelligence(
        self,
        asin: str,
        category_id: int,
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gather all network intelligence for an ASIN."""

        network_intel = {}

        # Get category benchmarks
        network_intel['category_benchmarks'] = self.network_intelligence.get_category_benchmarks(
            category_id=category_id,
            lookback_days=30
        )

        # Get competitive position
        network_intel['competitive_position'] = self.network_intelligence.get_competitive_position(
            asin=asin,
            category_id=category_id
        )

        # Get brand intelligence
        brand = current_metrics.get('brand', '')
        category_root = current_metrics.get('category_root', '')
        if brand and category_root:
            network_intel['brand_intelligence'] = self.network_intelligence.get_brand_intelligence(
                brand=brand,
                category_root=category_root
            )

        return network_intel

    def _calculate_historical_trends(self, df_historical: pd.DataFrame) -> Dict[str, Any]:
        """Calculate 30/90 day trends from historical data."""

        if df_historical.empty:
            return {}

        trends = {}

        # Price trend (30d)
        if 'price' in df_historical.columns and len(df_historical) > 720:  # 30 days hourly
            price_30d_ago = df_historical['price'].iloc[-720]
            price_now = df_historical['price'].iloc[-1]
            if price_30d_ago > 0:
                trends['price_change_30d'] = ((price_now - price_30d_ago) / price_30d_ago) * 100

        # BSR trend (30d)
        if 'bsr' in df_historical.columns and len(df_historical) > 720:
            bsr_30d_ago = df_historical['bsr'].iloc[-720]
            bsr_now = df_historical['bsr'].iloc[-1]
            if bsr_30d_ago > 0:
                trends['bsr_change_30d'] = ((bsr_now - bsr_30d_ago) / bsr_30d_ago) * 100

        # Review velocity (30d)
        if 'review_count' in df_historical.columns and len(df_historical) > 720:
            reviews_30d_ago = df_historical['review_count'].iloc[-720]
            reviews_now = df_historical['review_count'].iloc[-1]
            trends['reviews_gained_30d'] = reviews_now - reviews_30d_ago

        # BuyBox trend (30d)
        if 'buybox_share' in df_historical.columns and len(df_historical) > 720:
            bb_30d_ago = df_historical['buybox_share'].iloc[-720]
            bb_now = df_historical['buybox_share'].iloc[-1]
            trends['buybox_change_30d'] = (bb_now - bb_30d_ago) * 100

        return trends

    def _summarize_competitors(self, df_competitors: pd.DataFrame) -> Dict[str, Any]:
        """Summarize competitor data for AI classification."""

        if df_competitors.empty:
            return {}

        summary = {
            'competitor_count': len(df_competitors),
            'competitors_oos': 0,
            'price_vs_median': 0,
            'reviews_vs_median': 0
        }

        # Count OOS competitors
        if 'inventory_count' in df_competitors.columns:
            summary['competitors_oos'] = (df_competitors['inventory_count'] == 0).sum()

        # Price comparison (simplified)
        if 'buy_box_price' in df_competitors.columns and len(df_competitors) > 0:
            median_price = df_competitors['buy_box_price'].median()
            if median_price > 0 and 'buy_box_price' in df_competitors.columns:
                your_price = df_competitors['buy_box_price'].iloc[0]  # Assume first row is yours
                summary['price_vs_median'] = ((your_price / median_price - 1) * 100)

        return summary

    def _summarize_category_benchmarks(self, benchmarks: Dict[str, Any]) -> str:
        """Create human-readable summary of category benchmarks."""

        if not benchmarks:
            return "No benchmark data available"

        parts = []
        if benchmarks.get('median_price'):
            parts.append(f"Median Price: ${benchmarks['median_price']:.2f}")
        if benchmarks.get('median_review_count'):
            parts.append(f"Median Reviews: {benchmarks['median_review_count']:.0f}")
        if benchmarks.get('total_asins_tracked'):
            parts.append(f"ASINs Tracked: {benchmarks['total_asins_tracked']}")

        return ", ".join(parts) if parts else "Limited benchmark data"

    def _summarize_competitive_position(self, position: Dict[str, Any]) -> str:
        """Create human-readable summary of competitive position."""

        if not position:
            return "Position unknown"

        parts = []
        if 'price_vs_median' in position:
            parts.append(f"Price {position['price_vs_median']:+.1f}% vs median")
        if 'reviews_vs_median' in position:
            parts.append(f"Reviews {position['reviews_vs_median']:+.1f}% vs median")

        advantages = position.get('competitive_advantages', [])
        if advantages:
            parts.append(f"{len(advantages)} advantage(s)")

        return ", ".join(parts) if parts else "Neutral position"

    def _store_intelligence(self, intelligence: UnifiedIntelligence) -> None:
        """Store unified intelligence in database."""

        try:
            # Store main insight
            insight_data = {
                'asin': intelligence.asin,
                'generated_at': intelligence.timestamp.isoformat(),
                'product_status': intelligence.product_status.value,
                'strategic_state': intelligence.strategic_state,
                'confidence': intelligence.confidence,
                'reasoning': intelligence.reasoning,
                'recommendation': intelligence.recommendation,
                'action_type': intelligence.action_type,
                'projected_upside_monthly': intelligence.projected_upside_monthly,
                'downside_risk_monthly': intelligence.downside_risk_monthly,
                'net_expected_value': intelligence.net_expected_value,
                'thirty_day_risk': intelligence.thirty_day_risk,
                'thirty_day_growth': intelligence.thirty_day_growth,
                'time_horizon_days': intelligence.time_horizon_days,
                'primary_trigger_type': intelligence.primary_trigger
            }

            result = self.supabase.table('strategic_insights').insert(insight_data).execute()
            insight_id = result.data[0]['id'] if result.data else None

            # Store trigger events
            if insight_id and intelligence.trigger_events:
                trigger_data = []
                for trigger in intelligence.trigger_events[:10]:  # Top 10 triggers
                    trigger_data.append({
                        'insight_id': insight_id,
                        'asin': intelligence.asin,
                        'event_type': trigger.event_type,
                        'severity': trigger.severity,
                        'detected_at': trigger.detected_at.isoformat(),
                        'metric_name': trigger.metric_name,
                        'baseline_value': trigger.baseline_value,
                        'current_value': trigger.current_value,
                        'delta_pct': trigger.delta_pct,
                        'affected_asin': trigger.affected_asin,
                        'related_asin': trigger.related_asin
                    })

                if trigger_data:
                    self.supabase.table('trigger_events').insert(trigger_data).execute()

            print(f"  💾 Stored intelligence in database (ID: {insight_id})")

        except Exception as e:
            print(f"  ⚠️  Error storing intelligence: {str(e)}")


def get_active_insights_from_db(
    supabase: Client,
    user_id: Optional[str] = None,
    priority_threshold: int = 50
) -> List[Dict[str, Any]]:
    """
    Retrieve active insights from database for Action Queue.

    Args:
        supabase: Supabase client
        user_id: Optional user ID filter
        priority_threshold: Minimum priority (0-100)

    Returns:
        List of active insights sorted by priority
    """

    try:
        query = supabase.table('strategic_insights').select('*').eq('status', 'active')

        if user_id:
            query = query.eq('user_id', user_id)

        result = query.execute()

        # Filter by priority threshold
        insights = []
        for insight in result.data:
            status = ProductStatus[insight['product_status'].upper()]
            if status.priority >= priority_threshold:
                insights.append(insight)

        # Sort by priority (descending)
        insights.sort(key=lambda x: ProductStatus[x['product_status'].upper()].priority, reverse=True)

        return insights

    except Exception as e:
        print(f"⚠️  Error fetching insights: {str(e)}")
        return []
