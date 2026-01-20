"""
Network Intelligence Accumulator

Automatically accumulates data from every user search into the intelligence network.
This creates the network effect where the AI gets smarter as more users search products.
"""

from datetime import date, datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from supabase import Client


class NetworkIntelligenceAccumulator:
    """
    Accumulates data from every user search into the intelligence network.

    Called automatically after Phase 2 market discovery completes.
    Stores data in:
    - product_snapshots (with category metadata)
    - category_intelligence (benchmarks)
    - brand_intelligence (aggregates)
    - market_patterns (historical patterns)
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase

    def accumulate_search_data(
        self,
        market_snapshot: pd.DataFrame,
        category_id: int,
        category_name: str,
        category_tree: List[str],
        skip_snapshot_write: bool = False
    ) -> None:
        """
        Store market snapshot data and update intelligence aggregates.

        ENHANCEMENT 2.3: Added skip_snapshot_write parameter to avoid duplicate writes
        when cache_market_snapshot() has already written the product snapshots.

        Args:
            market_snapshot: DataFrame from Phase 2 discovery
            category_id: Keepa category ID
            category_name: Category name
            category_tree: Full category path ['Health', 'Sports Nutrition', 'Protein']
            skip_snapshot_write: If True, skip product snapshot write (already done by cache_market_snapshot)
        """

        category_root = category_tree[0] if category_tree else category_name
        today = date.today()

        print(f"📊 Accumulating data: {len(market_snapshot)} ASINs in {category_name}")

        # Step 1: Store product snapshots (skip if already written)
        # ENHANCEMENT 2.3: This step can be skipped to avoid duplicate writes
        if not skip_snapshot_write:
            self._store_product_snapshots(
                market_snapshot,
                category_id,
                category_name,
                category_tree,
                category_root,
                today
            )
        else:
            print(f"  ⏭️ Skipping product snapshot write (already cached)")

        # Step 2: Update category intelligence
        self._update_category_intelligence(
            market_snapshot,
            category_id,
            category_name,
            category_root,
            today
        )

        # Step 3: Update brand intelligence
        self._update_brand_intelligence(
            market_snapshot,
            category_root,
            today
        )

        # Step 4: Detect and store patterns
        self._detect_and_store_patterns(
            market_snapshot,
            category_root
        )

        print(f"✅ Data accumulation complete for {category_name}")

    def _store_product_snapshots(
        self,
        df: pd.DataFrame,
        category_id: int,
        category_name: str,
        category_tree: List[str],
        category_root: str,
        snapshot_date: date
    ) -> None:
        """Store individual product snapshots with category metadata."""

        records = []
        for _, row in df.iterrows():
            record = {
                'asin': row['asin'],
                'snapshot_date': str(snapshot_date),
                'buy_box_price': float(row['price']) if pd.notna(row.get('price')) else None,
                'sales_rank': int(row['bsr']) if pd.notna(row.get('bsr')) else None,
                'review_count': int(row.get('review_count', 0)) if pd.notna(row.get('review_count')) else None,
                'rating': float(row.get('rating', 0)) if pd.notna(row.get('rating')) else None,
                'estimated_weekly_revenue': float(row['revenue_proxy']) if pd.notna(row.get('revenue_proxy')) else None,
                'estimated_units': int(row.get('monthly_units', 0) / 4) if pd.notna(row.get('monthly_units')) else None,
                'title': row.get('title'),
                'brand': row.get('brand'),
                'main_image': row.get('main_image'),
                'category_id': category_id,
                'category_name': category_name,
                'category_tree': category_tree,
                'category_root': category_root,
                'source': 'keepa',
                'fetched_at': datetime.now().isoformat()
            }
            records.append(record)

        # Batch upsert (insert or update if exists)
        if records:
            try:
                self.supabase.table('product_snapshots').upsert(
                    records,
                    on_conflict='asin,snapshot_date'
                ).execute()
                print(f"  ✓ Stored {len(records)} product snapshots")
            except Exception as e:
                print(f"  ⚠️  Error storing snapshots: {str(e)}")

    def _update_category_intelligence(
        self,
        df: pd.DataFrame,
        category_id: int,
        category_name: str,
        category_root: str,
        snapshot_date: date
    ) -> None:
        """Calculate and store category-level benchmarks."""

        # Calculate aggregate metrics
        intelligence = {
            'category_id': category_id,
            'category_name': category_name,
            'category_root': category_root,
            'snapshot_date': str(snapshot_date),

            # Price benchmarks
            'median_price': float(df['price'].median()) if 'price' in df else None,
            'p75_price': float(df['price'].quantile(0.75)) if 'price' in df else None,
            'p25_price': float(df['price'].quantile(0.25)) if 'price' in df else None,
            'avg_price': float(df['price'].mean()) if 'price' in df else None,
            'price_volatility_score': float(df['price'].std()) if 'price' in df else None,

            # Quality benchmarks
            'median_rating': float(df['rating'].median()) if 'rating' in df and pd.notna(df['rating'].median()) else None,
            'avg_rating': float(df['rating'].mean()) if 'rating' in df and pd.notna(df['rating'].mean()) else None,
            'median_review_count': int(df['review_count'].median()) if 'review_count' in df and pd.notna(df['review_count'].median()) else None,
            'avg_review_count': int(df['review_count'].mean()) if 'review_count' in df and pd.notna(df['review_count'].mean()) else None,

            # Rank benchmarks
            'median_bsr': int(df['bsr'].median()) if 'bsr' in df and pd.notna(df['bsr'].median()) else None,
            'p25_bsr': int(df['bsr'].quantile(0.25)) if 'bsr' in df and pd.notna(df['bsr'].quantile(0.25)) else None,
            'p75_bsr': int(df['bsr'].quantile(0.75)) if 'bsr' in df and pd.notna(df['bsr'].quantile(0.75)) else None,

            # Market structure
            'total_asins_tracked': len(df),
            'total_weekly_revenue': float(df['revenue_proxy'].sum()) if 'revenue_proxy' in df else None,
            'median_weekly_revenue': float(df['revenue_proxy'].median()) if 'revenue_proxy' in df else None,

            # Data quality
            'data_quality': self._assess_data_quality(df),
            'snapshot_count': len(df),
            'last_updated': datetime.now().isoformat()
        }

        # Upsert category intelligence
        try:
            self.supabase.table('category_intelligence').upsert(
                intelligence,
                on_conflict='category_id,snapshot_date'
            ).execute()
            print(f"  ✓ Updated category intelligence: {intelligence['data_quality']} quality, {len(df)} ASINs")
        except Exception as e:
            print(f"  ⚠️  Error updating category intelligence: {str(e)}")

    def _update_brand_intelligence(
        self,
        df: pd.DataFrame,
        category_root: str,
        today: date
    ) -> None:
        """Update brand-level aggregates."""

        if 'brand' not in df.columns:
            return

        # Group by brand
        brand_groups = df.groupby('brand')

        brand_records = []
        for brand, group in brand_groups:
            if pd.isna(brand) or brand == '':
                continue

            record = {
                'brand': brand,
                'category_root': category_root,
                'total_asins_tracked': len(group),
                'avg_price': float(group['price'].mean()) if 'price' in group and pd.notna(group['price'].mean()) else None,
                'median_price': float(group['price'].median()) if 'price' in group and pd.notna(group['price'].median()) else None,
                'avg_rating': float(group['rating'].mean()) if 'rating' in group and pd.notna(group['rating'].mean()) else None,
                'avg_review_count': int(group['review_count'].mean()) if 'review_count' in group and pd.notna(group['review_count'].mean()) else None,
                'total_weekly_revenue': float(group['revenue_proxy'].sum()) if 'revenue_proxy' in group else None,
                'avg_weekly_revenue': float(group['revenue_proxy'].mean()) if 'revenue_proxy' in group else None,
                'first_seen': str(today),
                'last_updated': datetime.now().isoformat(),
                'snapshot_count': len(group)
            }
            brand_records.append(record)

        # Batch upsert brand intelligence
        if brand_records:
            try:
                for record in brand_records:
                    self.supabase.table('brand_intelligence').upsert(
                        record,
                        on_conflict='brand,category_root'
                    ).execute()
                print(f"  ✓ Updated {len(brand_records)} brand profiles")
            except Exception as e:
                print(f"  ⚠️  Error updating brand intelligence: {str(e)}")

    def _detect_and_store_patterns(
        self,
        df: pd.DataFrame,
        category_root: str
    ) -> None:
        """
        Detect interesting market patterns and store for future reference.

        Patterns detected:
        - Review advantage → Price premium
        - Low competition → Higher margins
        - (More patterns as we learn)
        """

        # Pattern 1: Review advantage → Price premium
        if 'review_count' in df.columns and 'price' in df.columns and len(df) > 10:
            try:
                high_review_threshold = df['review_count'].quantile(0.75)
                high_review_products = df[df['review_count'] >= high_review_threshold]
                low_review_products = df[df['review_count'] < high_review_threshold]

                if len(high_review_products) > 0 and len(low_review_products) > 0:
                    high_median = high_review_products['price'].median()
                    low_median = low_review_products['price'].median()

                    if low_median > 0:
                        price_premium = (high_median / low_median)

                        if price_premium > 1.3:  # 30%+ premium
                            self._store_pattern(
                                pattern_type='review_advantage_price_premium',
                                category_root=category_root,
                                trigger_conditions={
                                    'review_count': f'>{high_review_threshold:.0f}',
                                    'category_median': f'{df["review_count"].median():.0f}'
                                },
                                typical_outcome=f'Can command {(price_premium - 1) * 100:.0f}% price premium',
                                avg_revenue_impact=float(
                                    high_review_products['revenue_proxy'].mean() -
                                    low_review_products['revenue_proxy'].mean()
                                ) if 'revenue_proxy' in high_review_products else 0
                            )
            except Exception as e:
                print(f"  ⚠️  Error detecting review premium pattern: {str(e)}")

        # Pattern 2: More patterns can be added here as we learn

    def _store_pattern(
        self,
        pattern_type: str,
        category_root: str,
        trigger_conditions: Dict,
        typical_outcome: str,
        avg_revenue_impact: float
    ) -> None:
        """Store observed pattern in pattern library."""

        pattern = {
            'pattern_type': pattern_type,
            'category_root': category_root,
            'trigger_conditions': trigger_conditions,
            'typical_outcome': typical_outcome,
            'avg_revenue_impact': avg_revenue_impact,
            'observed_count': 1,
            'first_observed': str(date.today()),
            'last_observed': str(date.today()),
            'confidence_score': 0.5  # Low confidence on first observation
        }

        try:
            # Check if pattern already exists
            existing = self.supabase.table('market_patterns').select('*').eq(
                'pattern_type', pattern_type
            ).eq('category_root', category_root).execute()

            if existing.data and len(existing.data) > 0:
                # Increment observed count and update confidence
                pattern_id = existing.data[0]['id']
                new_count = existing.data[0]['observed_count'] + 1
                new_confidence = min(0.95, 0.5 + (new_count * 0.05))  # Cap at 95%

                self.supabase.table('market_patterns').update({
                    'observed_count': new_count,
                    'last_observed': str(date.today()),
                    'confidence_score': new_confidence
                }).eq('id', pattern_id).execute()
                print(f"  ✓ Updated pattern: {pattern_type} (observed {new_count}x)")
            else:
                # Store new pattern
                self.supabase.table('market_patterns').insert(pattern).execute()
                print(f"  ✓ Detected new pattern: {pattern_type}")
        except Exception as e:
            print(f"  ⚠️  Error storing pattern: {str(e)}")

    def _assess_data_quality(self, df: pd.DataFrame) -> str:
        """Assess data quality based on completeness."""
        if df.empty:
            return 'LOW'

        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))

        if missing_pct < 0.1:
            return 'HIGH'
        elif missing_pct < 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
