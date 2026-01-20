"""
Unified Intelligence Pipeline Orchestrator

Master pipeline that orchestrates all intelligence systems:
1. Trigger Detection → Identify market changes
2. Network Intelligence → Get category benchmarks
3. AI Classification (v2) → Strategic state
4. AI Insight Generation (v3) → Actionable recommendations
5. Database Storage → Store unified intelligence

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
from utils.ai_engine_v2 import TriggerAwareAIEngine, validate_classification_quality
from utils.ai_engine_v3 import NetworkAwareInsightEngine, validate_insight_quality, calculate_net_expected_value


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
        self.ai_classifier = TriggerAwareAIEngine(openai_api_key)
        self.ai_insight_engine = NetworkAwareInsightEngine(openai_api_key)

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

        This is the main orchestration logic:
        1. Detect trigger events
        2. Get network intelligence
        3. Classify strategic state (AI v2)
        4. Generate insight (AI v3)
        5. Combine into UnifiedIntelligence

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
            # STEP 1: Detect trigger events
            print(f"  🔍 Detecting trigger events...")
            trigger_events = detect_trigger_events(
                asin=asin,
                df_historical=historical_data,
                df_competitors=competitor_data,
                lookback_days=30
            )
            print(f"     Found {len(trigger_events)} trigger events")

            # STEP 2: Get network intelligence
            print(f"  🌐 Querying network intelligence...")
            category_id = category_context.get('category_id')
            network_intel = self._gather_network_intelligence(asin, category_id, current_metrics)

            # STEP 3: Calculate historical trends
            historical_trends = self._calculate_historical_trends(historical_data)

            # STEP 4: Classify strategic state (AI v2)
            print(f"  🤖 Classifying strategic state...")
            classification = self.ai_classifier.classify_strategic_state(
                asin=asin,
                current_metrics=current_metrics,
                historical_trends=historical_trends,
                trigger_events=trigger_events,
                competitor_data=self._summarize_competitors(competitor_data)
            )

            # Validate classification quality
            if not validate_classification_quality(classification):
                print(f"  ⚠️  Classification failed quality gates - retrying...")
                # Could retry here, but for now just proceed with warning
                classification['confidence'] = max(30, classification['confidence'] - 20)

            print(f"     Status: {classification['product_status'].value}")
            print(f"     Confidence: {classification['confidence']:.0f}%")

            # STEP 5: Generate actionable insight (AI v3)
            print(f"  💡 Generating actionable insight...")
            insight = self.ai_insight_engine.generate_insight(
                asin=asin,
                classification=classification,
                trigger_events=trigger_events,
                network_intelligence=network_intel,
                current_metrics=current_metrics
            )

            # Validate insight quality
            if not validate_insight_quality(insight):
                print(f"  ⚠️  Insight failed quality gates - using fallback...")
                insight['confidence'] = 30
                insight['recommendation'] = "Insufficient data quality for specific recommendation. Monitor metrics and wait for clearer signals."

            print(f"     Action: {insight['action_type']}")
            print(f"     Upside: ${insight['projected_upside_monthly']:.2f}/mo")
            print(f"     Downside: ${insight['downside_risk_monthly']:.2f}/mo")

            # STEP 6: Calculate net expected value
            net_ev = calculate_net_expected_value(insight)
            print(f"     Net EV: ${net_ev:+.2f}/mo")

            # STEP 7: Combine into UnifiedIntelligence
            unified = UnifiedIntelligence(
                # Identity
                asin=asin,
                timestamp=datetime.now(),

                # Trigger events
                trigger_events=trigger_events,
                primary_trigger=trigger_events[0].event_type if trigger_events else None,

                # Strategic classification
                product_status=classification['product_status'],
                strategic_state=classification['strategic_state'],
                confidence=min(classification['confidence'], insight['confidence']),  # Use lower confidence
                reasoning=classification['reasoning'],

                # Predictive intelligence
                thirty_day_risk=classification['risk_score'],
                thirty_day_growth=classification['growth_score'],
                net_expected_value=net_ev,

                # Actionable insight
                recommendation=insight['recommendation'],
                projected_upside_monthly=insight['projected_upside_monthly'],
                downside_risk_monthly=insight['downside_risk_monthly'],
                action_type=insight['action_type'],
                time_horizon_days=insight['time_horizon_days'],

                # Network context (metadata only)
                category_benchmarks_summary=self._summarize_category_benchmarks(
                    network_intel.get('category_benchmarks', {})
                ),
                competitive_position_summary=self._summarize_competitive_position(
                    network_intel.get('competitive_position', {})
                )
            )

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
