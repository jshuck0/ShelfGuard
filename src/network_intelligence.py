"""
Network Intelligence Query Layer

Provides access to accumulated network intelligence (benchmarks, patterns, competitive position).
Used by AI engine to enrich product analysis with category context.
"""

from typing import Dict, List, Optional, Any
from datetime import date, timedelta
from supabase import Client
import pandas as pd


class NetworkIntelligence:
    """
    Query layer for accessing accumulated network intelligence.

    Provides:
    - Category benchmarks (median price, reviews, BSR from all users)
    - Competitive position analysis (percentile rankings)
    - Brand intelligence (brand-level aggregates)
    - Historical patterns ("we've seen this before")
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase

    def get_category_benchmarks(
        self,
        category_id: int,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get category-level benchmarks from accumulated data.

        Returns:
            Dict with median price, review counts, BSR, data quality, etc.
        """

        try:
            # Query latest category intelligence
            result = self.supabase.table('category_intelligence').select('*').eq(
                'category_id', category_id
            ).order('snapshot_date', desc=True).limit(1).execute()

            if not result.data or len(result.data) == 0:
                return self._get_fallback_benchmarks(category_id)

            benchmarks = result.data[0]

            # Enrich with historical trends if available
            try:
                cutoff_date = (date.today() - timedelta(days=lookback_days)).isoformat()
                historical = self.supabase.table('category_intelligence').select(
                    'snapshot_date, median_price, median_bsr'
                ).eq('category_id', category_id).gte(
                    'snapshot_date', cutoff_date
                ).execute()

                if historical.data and len(historical.data) > 1:
                    prices = [h['median_price'] for h in historical.data if h.get('median_price')]
                    if len(prices) > 1:
                        benchmarks['price_trend_30d'] = (prices[-1] - prices[0]) / prices[0] if prices[0] else 0
            except:
                pass

            return benchmarks

        except Exception as e:
            print(f"⚠️  Error fetching category benchmarks: {str(e)}")
            return self._get_fallback_benchmarks(category_id)

    def get_competitive_position(
        self,
        asin: str,
        category_id: int
    ) -> Dict[str, Any]:
        """
        Compare ASIN to category benchmarks.

        Returns:
            Dict with percentile rankings, advantages, weaknesses
        """

        try:
            # Get product's latest snapshot
            product = self.supabase.table('product_snapshots').select('*').eq(
                'asin', asin
            ).order('snapshot_date', desc=True).limit(1).execute()

            if not product.data or len(product.data) == 0:
                return {}

            product_data = product.data[0]

            # Get category benchmarks
            benchmarks = self.get_category_benchmarks(category_id)

            # Calculate competitive position
            position = {
                'asin': asin,
                'price_vs_median': self._calc_vs_median(
                    product_data.get('buy_box_price'),
                    benchmarks.get('median_price')
                ),
                'reviews_vs_median': self._calc_vs_median(
                    product_data.get('review_count'),
                    benchmarks.get('median_review_count')
                ),
                'rating_vs_median': (
                    product_data.get('rating', 0) - benchmarks.get('median_rating', 0)
                ) if product_data.get('rating') and benchmarks.get('median_rating') else 0,

                # Percentile rankings
                'price_percentile': self._calculate_percentile(
                    product_data.get('buy_box_price'),
                    category_id,
                    'buy_box_price'
                ),
                'review_percentile': self._calculate_percentile(
                    product_data.get('review_count'),
                    category_id,
                    'review_count'
                ),

                # Strategic assessment
                'competitive_advantages': self._identify_advantages(product_data, benchmarks),
                'competitive_weaknesses': self._identify_weaknesses(product_data, benchmarks)
            }

            return position

        except Exception as e:
            print(f"⚠️  Error calculating competitive position: {str(e)}")
            return {}

    def get_brand_intelligence(
        self,
        brand: str,
        category_root: str
    ) -> Dict[str, Any]:
        """
        Get brand-level intelligence from accumulated data.

        Returns:
            Brand's average metrics, market share, product count
        """

        try:
            result = self.supabase.table('brand_intelligence').select('*').eq(
                'brand', brand
            ).eq('category_root', category_root).execute()

            if not result.data or len(result.data) == 0:
                return {
                    'brand': brand,
                    'data_available': False,
                    'message': 'No historical data for this brand yet'
                }

            data = result.data[0]
            data['data_available'] = True
            return data

        except Exception as e:
            print(f"⚠️  Error fetching brand intelligence: {str(e)}")
            return {'brand': brand, 'data_available': False}

    def get_historical_pattern(
        self,
        pattern_type: str,
        category_root: str
    ) -> Optional[Dict[str, Any]]:
        """
        Query historical patterns we've observed.

        Returns:
            Pattern data including success rate, typical outcome, confidence
        """

        try:
            result = self.supabase.table('market_patterns').select('*').eq(
                'pattern_type', pattern_type
            ).eq('category_root', category_root).order(
                'observed_count', desc=True
            ).limit(1).execute()

            if not result.data or len(result.data) == 0:
                return None

            return result.data[0]

        except Exception as e:
            print(f"⚠️  Error fetching pattern: {str(e)}")
            return None

    def get_similar_products(
        self,
        asin: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar products in our database.

        Uses price, category, and review count to find comparable ASINs.
        """

        try:
            # Get product's latest snapshot
            product = self.supabase.table('product_snapshots').select('*').eq(
                'asin', asin
            ).order('snapshot_date', desc=True).limit(1).execute()

            if not product.data or len(product.data) == 0:
                return []

            p = product.data[0]

            # Find similar products in same category with similar price
            price = p.get('buy_box_price', 0)
            if not price:
                return []

            price_range_min = price * 0.7
            price_range_max = price * 1.3

            similar = self.supabase.table('product_snapshots').select('*').eq(
                'category_id', p.get('category_id')
            ).gte('buy_box_price', price_range_min).lte(
                'buy_box_price', price_range_max
            ).neq('asin', asin).order(
                'snapshot_date', desc=True
            ).limit(limit).execute()

            return similar.data if similar.data else []

        except Exception as e:
            print(f"⚠️  Error finding similar products: {str(e)}")
            return []

    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================

    def _calculate_percentile(
        self,
        value: Optional[float],
        category_id: int,
        metric: str
    ) -> float:
        """Calculate percentile rank for a metric within category."""

        if value is None:
            return 50.0  # Default to median

        try:
            # Get all values for this metric in category
            products = self.supabase.table('product_snapshots').select(metric).eq(
                'category_id', category_id
            ).execute()

            if not products.data:
                return 50.0

            values = [p[metric] for p in products.data if p.get(metric) is not None]

            if not values or len(values) == 0:
                return 50.0

            # Calculate percentile
            rank = sum(1 for v in values if v < value)
            percentile = (rank / len(values)) * 100

            return percentile

        except Exception as e:
            print(f"⚠️  Error calculating percentile: {str(e)}")
            return 50.0

    def _calc_vs_median(
        self,
        value: Optional[float],
        median: Optional[float]
    ) -> float:
        """Calculate percentage vs median."""
        if value is None or median is None or median == 0:
            return 0.0
        return ((value / median - 1) * 100)

    def _identify_advantages(
        self,
        product: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> List[str]:
        """Identify competitive advantages."""
        advantages = []

        # Review advantage
        if product.get('review_count') and benchmarks.get('median_review_count'):
            if product['review_count'] > benchmarks['median_review_count'] * 1.5:
                advantages.append(
                    f"Review advantage: {product['review_count']} vs "
                    f"{int(benchmarks['median_review_count'])} median"
                )

        # Rating advantage
        if product.get('rating') and benchmarks.get('median_rating'):
            if product['rating'] > benchmarks['median_rating'] + 0.3:
                advantages.append(
                    f"Rating advantage: {product['rating']:.1f}★ vs "
                    f"{benchmarks['median_rating']:.1f}★ median"
                )

        # Price advantage (lower)
        if product.get('buy_box_price') and benchmarks.get('median_price'):
            if product['buy_box_price'] < benchmarks['median_price'] * 0.9:
                advantages.append(
                    f"Price advantage: ${product['buy_box_price']:.2f} "
                    f"(10%+ below median)"
                )

        return advantages

    def _identify_weaknesses(
        self,
        product: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> List[str]:
        """Identify competitive weaknesses."""
        weaknesses = []

        # Review gap
        if product.get('review_count') and benchmarks.get('median_review_count'):
            if product['review_count'] < benchmarks['median_review_count'] * 0.5:
                weaknesses.append(
                    f"Review gap: {product['review_count']} vs "
                    f"{int(benchmarks['median_review_count'])} median"
                )

        # Rating gap
        if product.get('rating') and benchmarks.get('median_rating'):
            if product['rating'] < benchmarks['median_rating'] - 0.3:
                weaknesses.append(
                    f"Rating gap: {product['rating']:.1f}★ vs "
                    f"{benchmarks['median_rating']:.1f}★ median"
                )

        return weaknesses

    def _get_fallback_benchmarks(self, category_id: int) -> Dict[str, Any]:
        """Fallback benchmarks when no data available."""
        return {
            'category_id': category_id,
            'median_price': None,
            'median_review_count': None,
            'median_rating': None,
            'median_bsr': None,
            'total_asins_tracked': 0,
            'data_quality': 'LOW',
            'message': 'No historical data available yet - be the first!'
        }
