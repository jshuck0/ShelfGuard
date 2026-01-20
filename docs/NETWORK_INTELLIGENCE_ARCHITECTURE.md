# ShelfGuard Network Intelligence Architecture
## Self-Learning AI System via Supabase Data Accumulation

**Status:** Architecture Complete - Ready for Implementation
**Author:** Technical Architecture Team
**Date:** 2026-01-19
**Related:** AI_PREDICTIVE_ENGINE_ARCHITECTURE.md, INSIGHT_ENGINE_REFACTOR_PLAN.md

---

## Executive Summary

**The Opportunity:**
Every time a user searches for a product or creates a project, we harvest Keepa data and store it in `product_snapshots`. This creates a **continuously growing intelligence network** where:
- More searches = More category-level benchmarks
- More users = Better competitive intelligence
- More time = Richer historical pattern recognition

**The Network Effect:**
```
User 1 searches "protein powder" → We learn category norms for Sports Nutrition
User 2 searches "Quest bars" → AI now has competitive context for User 1's products
User 3 searches "Optimum Nutrition" → Pattern recognition improves for everyone
```

**Key Insight:** Instead of analyzing products in isolation, the AI leverages **the entire corpus of stored data** to provide:
1. **Category Intelligence** - What's "normal" for this category?
2. **Competitive Benchmarking** - How does this product compare to similar ASINs we've seen?
3. **Historical Pattern Recognition** - Have we seen this pattern before? What happened?
4. **Cross-Product Learning** - Insights from one product inform recommendations for similar products

---

## Part 1: Data Accumulation Strategy

### 1.1 Current Data Storage (Existing)

**`product_snapshots` Table** - The Central Intelligence Repository:
```sql
CREATE TABLE product_snapshots (
    id UUID PRIMARY KEY,
    asin TEXT NOT NULL,
    snapshot_date DATE NOT NULL,

    -- Pricing signals
    buy_box_price NUMERIC(10, 2),
    amazon_price NUMERIC(10, 2),
    new_fba_price NUMERIC(10, 2),

    -- Competitive signals
    sales_rank INTEGER,
    amazon_bb_share NUMERIC(5, 4),
    buy_box_switches INTEGER,
    new_offer_count INTEGER,

    -- Social proof
    review_count INTEGER,
    rating NUMERIC(3, 2),

    -- Performance
    estimated_units INTEGER,
    estimated_weekly_revenue NUMERIC(12, 2),

    -- Metadata
    title TEXT,
    brand TEXT,
    parent_asin TEXT,
    main_image TEXT,
    fetched_at TIMESTAMP,

    UNIQUE(asin, snapshot_date)
);
```

**Growth Model:**
- Day 1: 100 ASINs searched → 100 snapshots
- Day 30: 1,000 users × 20 ASINs each → 20,000 snapshots
- Day 365: Continuous growth → 500,000+ snapshots across all categories

### 1.2 Enhanced Schema for Network Intelligence

**Add Category Tagging:**
```sql
-- Extend product_snapshots with category metadata
ALTER TABLE product_snapshots
ADD COLUMN category_id INTEGER,
ADD COLUMN category_name TEXT,
ADD COLUMN category_tree TEXT[],  -- Array: ['Health & Household', 'Sports Nutrition', 'Protein Bars']
ADD COLUMN category_root TEXT;    -- Root: 'Health & Household'

CREATE INDEX idx_snapshots_category ON product_snapshots(category_id, snapshot_date DESC);
CREATE INDEX idx_snapshots_category_root ON product_snapshots(category_root, snapshot_date DESC);
```

**Add Brand Intelligence:**
```sql
-- Track brand-level aggregates
CREATE TABLE brand_intelligence (
    brand TEXT PRIMARY KEY,
    category_root TEXT,

    -- Aggregate metrics (updated daily)
    total_asins_tracked INTEGER DEFAULT 0,
    avg_price NUMERIC(10, 2),
    avg_rating NUMERIC(3, 2),
    avg_review_count INTEGER,
    total_weekly_revenue NUMERIC(12, 2),
    market_share_pct NUMERIC(5, 2),

    -- Time series
    first_seen DATE,
    last_updated TIMESTAMP DEFAULT NOW(),

    -- Metadata
    snapshot_count INTEGER DEFAULT 0  -- How many data points we have
);

CREATE INDEX idx_brand_intelligence_category ON brand_intelligence(category_root);
```

**Add Category Intelligence:**
```sql
-- Category-level benchmarks (updated daily)
CREATE TABLE category_intelligence (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL,
    category_root TEXT,
    snapshot_date DATE NOT NULL,

    -- Competitive benchmarks
    median_price NUMERIC(10, 2),
    p75_price NUMERIC(10, 2),
    p25_price NUMERIC(10, 2),
    median_rating NUMERIC(3, 2),
    median_review_count INTEGER,
    median_bsr INTEGER,

    -- Market dynamics
    total_asins_tracked INTEGER,
    avg_offer_count NUMERIC(5, 2),
    avg_bb_share NUMERIC(5, 4),
    price_volatility_score NUMERIC(5, 2),  -- Std dev of prices

    -- Intelligence quality
    data_quality TEXT CHECK (data_quality IN ('HIGH', 'MEDIUM', 'LOW')),
    snapshot_count INTEGER,  -- How many ASINs we have data for

    UNIQUE(category_id, snapshot_date)
);

CREATE INDEX idx_category_intelligence_date ON category_intelligence(snapshot_date DESC);
CREATE INDEX idx_category_intelligence_root ON category_intelligence(category_root, snapshot_date DESC);
```

**Add Historical Pattern Library:**
```sql
-- Pattern recognition: store known patterns we've observed
CREATE TABLE market_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type TEXT NOT NULL,  -- 'competitor_oos_price_spike', 'seasonal_dip', 'review_velocity_conversion'
    category_root TEXT,

    -- Pattern signature (what triggers it)
    trigger_conditions JSONB NOT NULL,  -- {"competitor_inventory": "<5", "review_advantage": ">2x"}

    -- Historical outcomes (what usually happens)
    typical_outcome TEXT,
    success_rate NUMERIC(5, 2),  -- % of times pattern leads to predicted outcome
    avg_revenue_impact NUMERIC(10, 2),
    avg_duration_days INTEGER,

    -- Sample data
    observed_count INTEGER DEFAULT 1,
    first_observed DATE,
    last_observed DATE,
    example_asins TEXT[],  -- Sample ASINs where we saw this

    -- Metadata
    confidence_score NUMERIC(3, 2),  -- 0.0 - 1.0
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_patterns_type_category ON market_patterns(pattern_type, category_root);
CREATE INDEX idx_patterns_observed_count ON market_patterns(observed_count DESC);
```

### 1.3 Data Accumulation Pipeline

**Automatic Enrichment on Every Search:**

```python
# File: src/data_accumulation.py

from datetime import date
from typing import Dict, List
import pandas as pd
from supabase import Client

class NetworkIntelligenceAccumulator:
    """
    Accumulates data from every user search into the intelligence network.

    Called automatically after Phase 2 market discovery completes.
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase

    def accumulate_search_data(
        self,
        market_snapshot: pd.DataFrame,
        category_id: int,
        category_name: str,
        category_tree: List[str]
    ) -> None:
        """
        Store market snapshot data and update intelligence aggregates.

        Args:
            market_snapshot: DataFrame from Phase 2 discovery
            category_id: Keepa category ID
            category_name: Category name
            category_tree: Full category path ['Health', 'Sports Nutrition', 'Protein']
        """

        category_root = category_tree[0] if category_tree else category_name
        today = date.today()

        # ========== STEP 1: STORE PRODUCT SNAPSHOTS ==========
        self._store_product_snapshots(
            market_snapshot,
            category_id,
            category_name,
            category_tree,
            category_root,
            today
        )

        # ========== STEP 2: UPDATE CATEGORY INTELLIGENCE ==========
        self._update_category_intelligence(
            market_snapshot,
            category_id,
            category_name,
            category_root,
            today
        )

        # ========== STEP 3: UPDATE BRAND INTELLIGENCE ==========
        self._update_brand_intelligence(
            market_snapshot,
            category_root,
            today
        )

        # ========== STEP 4: DETECT AND STORE PATTERNS ==========
        self._detect_and_store_patterns(
            market_snapshot,
            category_root
        )

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
                'snapshot_date': snapshot_date,
                'buy_box_price': float(row['price']) if pd.notna(row['price']) else None,
                'sales_rank': int(row['bsr']) if pd.notna(row['bsr']) else None,
                'review_count': int(row.get('review_count', 0)),
                'rating': float(row.get('rating', 0)),
                'estimated_weekly_revenue': float(row['revenue_proxy']) if pd.notna(row['revenue_proxy']) else None,
                'estimated_units': int(row['monthly_units'] / 4) if pd.notna(row.get('monthly_units')) else None,
                'title': row.get('title'),
                'brand': row.get('brand'),
                'main_image': row.get('main_image'),
                'category_id': category_id,
                'category_name': category_name,
                'category_tree': category_tree,
                'category_root': category_root,
                'source': 'keepa'
            }
            records.append(record)

        # Upsert (insert or update if exists)
        self.supabase.table('product_snapshots').upsert(
            records,
            on_conflict='asin,snapshot_date'
        ).execute()

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
            'snapshot_date': snapshot_date,

            # Price benchmarks
            'median_price': float(df['price'].median()),
            'p75_price': float(df['price'].quantile(0.75)),
            'p25_price': float(df['price'].quantile(0.25)),

            # Quality benchmarks
            'median_rating': float(df['rating'].median()) if 'rating' in df else None,
            'median_review_count': int(df['review_count'].median()) if 'review_count' in df else None,
            'median_bsr': int(df['bsr'].median()) if pd.notna(df['bsr'].median()) else None,

            # Market structure
            'total_asins_tracked': len(df),
            'price_volatility_score': float(df['price'].std()),

            # Data quality
            'data_quality': self._assess_data_quality(df),
            'snapshot_count': len(df)
        }

        # Upsert category intelligence
        self.supabase.table('category_intelligence').upsert(
            intelligence,
            on_conflict='category_id,snapshot_date'
        ).execute()

    def _update_brand_intelligence(
        self,
        df: pd.DataFrame,
        category_root: str,
        today: date
    ) -> None:
        """Update brand-level aggregates."""

        # Group by brand
        brand_groups = df.groupby('brand')

        brand_records = []
        for brand, group in brand_groups:
            record = {
                'brand': brand,
                'category_root': category_root,
                'total_asins_tracked': len(group),
                'avg_price': float(group['price'].mean()),
                'avg_rating': float(group['rating'].mean()) if 'rating' in group else None,
                'avg_review_count': int(group['review_count'].mean()) if 'review_count' in group else None,
                'total_weekly_revenue': float(group['revenue_proxy'].sum()),
                'last_updated': today,
                'snapshot_count': len(group)
            }
            brand_records.append(record)

        # Upsert brand intelligence
        for record in brand_records:
            self.supabase.table('brand_intelligence').upsert(
                record,
                on_conflict='brand,category_root'
            ).execute()

    def _detect_and_store_patterns(
        self,
        df: pd.DataFrame,
        category_root: str
    ) -> None:
        """
        Detect interesting market patterns and store for future reference.

        Examples:
        - High-review products command 2x price premium
        - Products with <5 competitors have 3x margin
        - Review velocity >10/week correlates with BSR improvement
        """

        # Pattern 1: Review advantage → Price premium
        if 'review_count' in df.columns and len(df) > 10:
            high_review_threshold = df['review_count'].quantile(0.75)
            high_review_products = df[df['review_count'] >= high_review_threshold]
            low_review_products = df[df['review_count'] < high_review_threshold]

            if len(high_review_products) > 0 and len(low_review_products) > 0:
                price_premium = (
                    high_review_products['price'].median() /
                    low_review_products['price'].median()
                )

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
                        )
                    )

        # Pattern 2: Competitor stockout → Price lift opportunity
        # (Will detect when we have inventory data)

        # Pattern 3: More patterns as we learn...

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
            'first_observed': date.today(),
            'last_observed': date.today(),
            'confidence_score': 0.5  # Low confidence on first observation
        }

        # Check if pattern already exists
        existing = self.supabase.table('market_patterns').select('*').eq(
            'pattern_type', pattern_type
        ).eq('category_root', category_root).execute()

        if existing.data:
            # Increment observed count and update confidence
            pattern_id = existing.data[0]['id']
            new_count = existing.data[0]['observed_count'] + 1
            new_confidence = min(0.95, 0.5 + (new_count * 0.05))  # Cap at 95%

            self.supabase.table('market_patterns').update({
                'observed_count': new_count,
                'last_observed': date.today(),
                'confidence_score': new_confidence
            }).eq('id', pattern_id).execute()
        else:
            # Store new pattern
            self.supabase.table('market_patterns').insert(pattern).execute()

    def _assess_data_quality(self, df: pd.DataFrame) -> str:
        """Assess data quality based on completeness."""
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))

        if missing_pct < 0.1:
            return 'HIGH'
        elif missing_pct < 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
```

---

## Part 2: Network Intelligence Queries

### 2.1 Category Benchmarking

**Query: "How does this product compare to category norms?"**

```python
# File: src/network_intelligence.py

class NetworkIntelligence:
    """
    Query layer for accessing accumulated network intelligence.

    Used by AI engine to enrich product analysis with category context.
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase

    def get_category_benchmarks(
        self,
        category_id: int,
        lookback_days: int = 30
    ) -> Dict:
        """
        Get category-level benchmarks from accumulated data.

        Returns:
            Dict with median price, review counts, BSR, etc.
        """

        # Query latest category intelligence
        result = self.supabase.table('category_intelligence').select('*').eq(
            'category_id', category_id
        ).order('snapshot_date', desc=True).limit(1).execute()

        if not result.data:
            return self._get_fallback_benchmarks(category_id)

        benchmarks = result.data[0]

        # Enrich with historical trends
        historical = self.supabase.table('category_intelligence').select(
            'snapshot_date, median_price, median_bsr'
        ).eq('category_id', category_id).gte(
            'snapshot_date', date.today() - timedelta(days=lookback_days)
        ).execute()

        if historical.data:
            prices = [h['median_price'] for h in historical.data if h['median_price']]
            benchmarks['price_trend_30d'] = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0

        return benchmarks

    def get_competitive_position(
        self,
        asin: str,
        category_id: int
    ) -> Dict:
        """
        Compare ASIN to category benchmarks.

        Returns:
            Dict with percentile rankings, advantages, weaknesses
        """

        # Get product's latest snapshot
        product = self.supabase.table('product_snapshots').select('*').eq(
            'asin', asin
        ).order('snapshot_date', desc=True).limit(1).execute()

        if not product.data:
            return {}

        product_data = product.data[0]

        # Get category benchmarks
        benchmarks = self.get_category_benchmarks(category_id)

        # Calculate competitive position
        position = {
            'asin': asin,
            'price_vs_median': (
                (product_data['buy_box_price'] / benchmarks['median_price'] - 1) * 100
            ) if product_data['buy_box_price'] and benchmarks['median_price'] else 0,

            'reviews_vs_median': (
                (product_data['review_count'] / benchmarks['median_review_count'] - 1) * 100
            ) if product_data['review_count'] and benchmarks['median_review_count'] else 0,

            'rating_vs_median': (
                product_data['rating'] - benchmarks['median_rating']
            ) if product_data['rating'] and benchmarks['median_rating'] else 0,

            # Percentile rankings
            'price_percentile': self._calculate_percentile(
                product_data['buy_box_price'],
                category_id,
                'buy_box_price'
            ),

            'review_percentile': self._calculate_percentile(
                product_data['review_count'],
                category_id,
                'review_count'
            ),

            # Strategic assessment
            'competitive_advantages': self._identify_advantages(product_data, benchmarks),
            'competitive_weaknesses': self._identify_weaknesses(product_data, benchmarks)
        }

        return position

    def get_brand_intelligence(
        self,
        brand: str,
        category_root: str
    ) -> Dict:
        """
        Get brand-level intelligence from accumulated data.

        Returns:
            Brand's average metrics, market share, product count
        """

        result = self.supabase.table('brand_intelligence').select('*').eq(
            'brand', brand
        ).eq('category_root', category_root).execute()

        if not result.data:
            return {
                'brand': brand,
                'data_available': False,
                'message': 'No historical data for this brand yet'
            }

        return result.data[0]

    def get_similar_products(
        self,
        asin: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find similar products in our database.

        Uses price, category, and review count to find comparable ASINs.
        """

        # Get product's latest snapshot
        product = self.supabase.table('product_snapshots').select('*').eq(
            'asin', asin
        ).order('snapshot_date', desc=True).limit(1).execute()

        if not product.data:
            return []

        p = product.data[0]

        # Find similar products in same category with similar price
        price_range_min = p['buy_box_price'] * 0.7 if p['buy_box_price'] else 0
        price_range_max = p['buy_box_price'] * 1.3 if p['buy_box_price'] else 999999

        similar = self.supabase.table('product_snapshots').select('*').eq(
            'category_id', p['category_id']
        ).gte('buy_box_price', price_range_min).lte(
            'buy_box_price', price_range_max
        ).neq('asin', asin).order(
            'snapshot_date', desc=True
        ).limit(limit).execute()

        return similar.data

    def get_historical_pattern(
        self,
        pattern_type: str,
        category_root: str
    ) -> Dict:
        """
        Query historical patterns we've observed.

        Returns:
            Pattern data including success rate, typical outcome, confidence
        """

        result = self.supabase.table('market_patterns').select('*').eq(
            'pattern_type', pattern_type
        ).eq('category_root', category_root).order(
            'observed_count', desc=True
        ).limit(1).execute()

        if not result.data:
            return None

        return result.data[0]

    def _calculate_percentile(
        self,
        value: float,
        category_id: int,
        metric: str
    ) -> float:
        """Calculate percentile rank for a metric within category."""

        # Get all values for this metric in category
        products = self.supabase.table('product_snapshots').select(metric).eq(
            'category_id', category_id
        ).execute()

        if not products.data:
            return 50.0  # Default to median

        values = [p[metric] for p in products.data if p[metric] is not None]

        if not values:
            return 50.0

        # Calculate percentile
        rank = sum(1 for v in values if v < value)
        percentile = (rank / len(values)) * 100

        return percentile

    def _identify_advantages(self, product: Dict, benchmarks: Dict) -> List[str]:
        """Identify competitive advantages."""
        advantages = []

        if product['review_count'] > benchmarks['median_review_count'] * 1.5:
            advantages.append(f"Review advantage: {product['review_count']} vs {benchmarks['median_review_count']:.0f} median")

        if product['rating'] > benchmarks['median_rating'] + 0.3:
            advantages.append(f"Rating advantage: {product['rating']:.1f}★ vs {benchmarks['median_rating']:.1f}★ median")

        if product['buy_box_price'] < benchmarks['median_price'] * 0.9:
            advantages.append(f"Price advantage: ${product['buy_box_price']:.2f} (10%+ below median)")

        return advantages

    def _identify_weaknesses(self, product: Dict, benchmarks: Dict) -> List[str]:
        """Identify competitive weaknesses."""
        weaknesses = []

        if product['review_count'] < benchmarks['median_review_count'] * 0.5:
            weaknesses.append(f"Review gap: {product['review_count']} vs {benchmarks['median_review_count']:.0f} median")

        if product['rating'] < benchmarks['median_rating'] - 0.3:
            weaknesses.append(f"Rating gap: {product['rating']:.1f}★ vs {benchmarks['median_rating']:.1f}★ median")

        return weaknesses
```

---

## Part 3: AI Engine Integration

### 3.1 Enhanced LLM Prompts with Network Intelligence

**Inject accumulated data into strategic classification:**

```python
# File: utils/ai_engine_v3.py (Network-Aware)

async def analyze_product_with_network_intelligence(
    asin: str,
    row_data: Dict[str, Any],
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame,
    category_id: int,
    client: Optional[AsyncOpenAI] = None,
    strategic_bias: str = "Balanced Defense"
) -> UnifiedIntelligence:
    """
    NETWORK-AWARE INTELLIGENCE PIPELINE

    Enhancement: Now leverages accumulated data from all users' searches.

    NEW INPUTS:
    - Category benchmarks from network
    - Brand intelligence from network
    - Historical patterns from network
    - Similar product performance data
    """

    # Initialize network intelligence client
    supabase = get_supabase_client()
    network_intel = NetworkIntelligence(supabase)

    # ========== STEP 1: TRIGGER EVENT DETECTION (EXISTING) ==========
    trigger_events = detect_trigger_events(
        asin=asin,
        df_historical=df_historical,
        df_competitors=df_competitors
    )

    # ========== STEP 2: NETWORK INTELLIGENCE ENRICHMENT (NEW) ==========

    # Get category benchmarks from accumulated data
    category_benchmarks = network_intel.get_category_benchmarks(category_id)

    # Get competitive position analysis
    competitive_position = network_intel.get_competitive_position(asin, category_id)

    # Get brand intelligence
    brand = row_data.get('brand', 'Unknown')
    category_root = category_benchmarks.get('category_root', '')
    brand_intel = network_intel.get_brand_intelligence(brand, category_root)

    # Get similar products for comparison
    similar_products = network_intel.get_similar_products(asin, limit=5)

    # Query historical patterns
    relevant_patterns = []
    for event in trigger_events:
        pattern = network_intel.get_historical_pattern(
            pattern_type=event.event_type,
            category_root=category_root
        )
        if pattern:
            relevant_patterns.append(pattern)

    # ========== STEP 3: BUILD ENHANCED CONTEXT ==========

    network_context = {
        'category_benchmarks': category_benchmarks,
        'competitive_position': competitive_position,
        'brand_intelligence': brand_intel,
        'similar_products': similar_products,
        'historical_patterns': relevant_patterns,

        # Meta-intelligence (how much data do we have?)
        'data_quality': category_benchmarks.get('data_quality', 'LOW'),
        'asins_in_category': category_benchmarks.get('total_asins_tracked', 0),
        'confidence_boost': calculate_confidence_boost(category_benchmarks)
    }

    # ========== STEP 4: ENHANCED LLM PROMPT (NEW) ==========

    prompt = build_network_aware_prompt(
        row_data=row_data,
        trigger_events=trigger_events,
        network_context=network_context
    )

    # Call LLM with enriched context
    # ... (rest of existing logic)

    return intelligence


def build_network_aware_prompt(
    row_data: Dict[str, Any],
    trigger_events: List[TriggerEvent],
    network_context: Dict[str, Any]
) -> str:
    """
    Build LLM prompt enriched with network intelligence.

    NEW: Includes category benchmarks, historical patterns, similar product data.
    """

    benchmarks = network_context['category_benchmarks']
    position = network_context['competitive_position']
    brand_intel = network_context['brand_intelligence']
    patterns = network_context['historical_patterns']

    # Format network intelligence section
    network_section = f"""
NETWORK INTELLIGENCE (Based on {benchmarks.get('total_asins_tracked', 0)} ASINs in category):
Data Quality: {network_context['data_quality']} ({benchmarks.get('snapshot_count', 0)} snapshots)

CATEGORY BENCHMARKS:
- Median Price: ${benchmarks.get('median_price', 0):.2f}
- P75 Price (Premium): ${benchmarks.get('p75_price', 0):.2f}
- P25 Price (Budget): ${benchmarks.get('p25_price', 0):.2f}
- Median Review Count: {benchmarks.get('median_review_count', 0)}
- Median Rating: {benchmarks.get('median_rating', 0):.1f}★
- Median BSR: {benchmarks.get('median_bsr', 0):,}
- Price Volatility: {benchmarks.get('price_volatility_score', 0):.2f} (higher = more volatile)

YOUR COMPETITIVE POSITION:
- Price vs Median: {position.get('price_vs_median', 0):+.1f}% ({"PREMIUM" if position.get('price_vs_median', 0) > 0 else "DISCOUNT"})
- Reviews vs Median: {position.get('reviews_vs_median', 0):+.1f}%
- Rating vs Median: {position.get('rating_vs_median', 0):+.1f} stars
- Price Percentile: {position.get('price_percentile', 50):.0f}th percentile
- Review Percentile: {position.get('review_percentile', 50):.0f}th percentile

COMPETITIVE ADVANTAGES:
{chr(10).join(['✓ ' + adv for adv in position.get('competitive_advantages', [])])}

COMPETITIVE WEAKNESSES:
{chr(10).join(['✗ ' + weak for weak in position.get('competitive_weaknesses', [])])}

BRAND INTELLIGENCE ({brand_intel.get('brand', 'Unknown')}):
{f"- Tracked ASINs: {brand_intel.get('total_asins_tracked', 0)}" if brand_intel.get('data_available') else "- No historical brand data available (first time seeing this brand)"}
{f"- Avg Price: ${brand_intel.get('avg_price', 0):.2f}" if brand_intel.get('data_available') else ""}
{f"- Avg Rating: {brand_intel.get('avg_rating', 0):.1f}★" if brand_intel.get('data_available') else ""}
{f"- Weekly Revenue: ${brand_intel.get('total_weekly_revenue', 0):,.0f}" if brand_intel.get('data_available') else ""}

HISTORICAL PATTERNS (We've seen this before):
{chr(10).join([
    f"- {p['pattern_type']}: {p['typical_outcome']} "
    f"(Observed {p['observed_count']}x, {p['confidence_score']:.0%} confidence)"
    for p in patterns
]) if patterns else "- No historical patterns match current triggers"}
"""

    # Build full prompt (combine with existing trigger event section)
    trigger_section = "\n".join([e.to_llm_context() for e in trigger_events[:5]])

    prompt = f"""You are ShelfGuard's Senior CPG Strategist analyzing ASIN {row_data['asin']}.

PRODUCT CURRENT STATE:
- Monthly Revenue: ${row_data['monthly_revenue']:,.0f}
- Current Price: ${row_data['price']:.2f}
- Sales Rank (BSR): {row_data['bsr']:,}
- Review Count: {row_data['review_count']} ({row_data.get('avg_rating', 0):.1f}★)
- Brand: {row_data.get('brand', 'Unknown')}

{network_section}

DETECTED TRIGGER EVENTS (Last 30 Days):
{trigger_section if trigger_events else "No significant trigger events detected."}

TASK: Classify this product and generate strategic recommendation.

## Critical Rules

1. **Leverage Network Intelligence**: Use category benchmarks and historical patterns to inform your analysis
2. **Cite Specific Data**: Reference percentiles, competitive advantages, and historical patterns
3. **Calibrate Confidence**: Higher data quality ({network_context['data_quality']}) and more patterns observed = higher confidence
4. **Historical Validation**: If we've seen similar patterns before, cite the success rate and typical outcome

## Example Reasoning (WITH Network Intelligence)

❌ BAD: "Product is performing well in competitive market"
✅ GOOD: "Product is in 75th percentile for reviews (450 vs 220 median) and commands 15% price premium
($26.99 vs $23.50 median). Historical pattern 'review_advantage_price_premium' observed 12x in this
category with 85% confidence suggests products with 2x review advantage can sustain 20-30% premiums."

Generate the classification now (JSON format).
"""

    return prompt


def calculate_confidence_boost(benchmarks: Dict) -> float:
    """
    Calculate confidence boost based on data quality.

    More data = higher confidence in predictions.
    """
    asins_tracked = benchmarks.get('total_asins_tracked', 0)
    data_quality = benchmarks.get('data_quality', 'LOW')

    # Base confidence boost
    if data_quality == 'HIGH' and asins_tracked > 50:
        return 0.15  # +15% confidence
    elif data_quality == 'MEDIUM' and asins_tracked > 20:
        return 0.10  # +10% confidence
    elif asins_tracked > 10:
        return 0.05  # +5% confidence
    else:
        return 0.0  # No boost (not enough data)
```

---

## Part 4: Implementation Roadmap

### Phase 1: Schema & Data Accumulation (Week 1)

**Day 1-2: Schema Extension**
- [ ] Add category/brand columns to `product_snapshots`
- [ ] Create `category_intelligence` table
- [ ] Create `brand_intelligence` table
- [ ] Create `market_patterns` table

**Day 3-4: Accumulation Pipeline**
- [ ] Create `src/data_accumulation.py`
- [ ] Implement `NetworkIntelligenceAccumulator`
- [ ] Hook into Phase 2 discovery completion

**Day 5: Network Query Layer**
- [ ] Create `src/network_intelligence.py`
- [ ] Implement benchmark queries
- [ ] Implement competitive position analysis

### Phase 2: AI Integration (Week 2)

**Day 1-2: Enhanced Prompts**
- [ ] Create `utils/ai_engine_v3.py` (network-aware)
- [ ] Implement `build_network_aware_prompt()`
- [ ] Add network context injection

**Day 3-4: Pattern Recognition**
- [ ] Implement pattern detection logic
- [ ] Store observed patterns
- [ ] Query and cite patterns in LLM prompts

**Day 5: Testing**
- [ ] Test with real category data
- [ ] Verify confidence boost logic
- [ ] Validate pattern storage

### Phase 3: UI Integration (Week 3)

**Day 1-2: Dashboard Enrichment**
- [ ] Show category benchmarks in UI
- [ ] Display competitive position chart
- [ ] Show network intelligence quality indicator

**Day 3-4: Pattern Library UI**
- [ ] Build pattern explorer view
- [ ] Show "We've seen this before" indicators
- [ ] Display success rates

**Day 5: Analytics**
- [ ] Track network effect metrics
- [ ] Show data accumulation growth
- [ ] Display prediction accuracy improvements

---

## Part 5: Network Effect Metrics

### Growth Tracking

```sql
-- Query: Network growth over time
SELECT
    snapshot_date,
    COUNT(DISTINCT asin) as unique_asins,
    COUNT(DISTINCT category_id) as unique_categories,
    COUNT(*) as total_snapshots
FROM product_snapshots
GROUP BY snapshot_date
ORDER BY snapshot_date DESC
LIMIT 90;

-- Query: Data quality by category
SELECT
    category_root,
    COUNT(DISTINCT asin) as asins_tracked,
    AVG(snapshot_count) as avg_snapshots_per_asin,
    MAX(snapshot_date) as last_updated,
    data_quality
FROM category_intelligence
GROUP BY category_root, data_quality
ORDER BY asins_tracked DESC;

-- Query: Most valuable patterns
SELECT
    pattern_type,
    category_root,
    observed_count,
    confidence_score,
    success_rate,
    avg_revenue_impact
FROM market_patterns
WHERE confidence_score > 0.7
ORDER BY observed_count DESC
LIMIT 20;
```

### Success Metrics

| Metric | Day 1 | Month 1 | Month 6 | Goal |
|--------|-------|---------|---------|------|
| **Total ASINs** | 100 | 20,000 | 500,000 | 1M+ |
| **Categories Covered** | 5 | 50 | 200 | 300+ |
| **Data Quality (HIGH)** | 10% | 30% | 60% | 80%+ |
| **Patterns Discovered** | 0 | 25 | 150 | 500+ |
| **Prediction Confidence** | 65% | 70% | 80% | 85%+ |

---

## Part 6: The Flywheel Effect

```
┌─────────────────────────────────────────────────────────────┐
│                    NETWORK FLYWHEEL                         │
└─────────────────────────────────────────────────────────────┘

  User Searches Product
         ↓
  Data Stored in Network
         ↓
  Category Benchmarks Improve
         ↓
  AI Gets Better Context
         ↓
  Predictions More Accurate
         ↓
  Users Trust Recommendations
         ↓
  More Users Search More Products ← LOOP
```

**Example:**
- **Week 1**: User A searches "protein powder" → We have 100 ASINs, predictions at 65% accuracy
- **Week 10**: 50 users searched protein category → We have 2,000 ASINs, predictions at 75% accuracy
- **Week 50**: 500 users → We have 20,000 ASINs, know historical patterns, predictions at 85% accuracy
- **Result**: Last user gets FAR better insights than first user, without any code changes

---

**End of Network Intelligence Architecture. Ready for Implementation.**
