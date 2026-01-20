# Staged Intelligence Features

The following modules are **BUILT but NOT YET ACTIVATED** in the main dashboard.
These represent the advanced intelligence pipeline that was recently refactored and is staged for future activation without disrupting current flows.

---

## Intelligence Pipeline (`src/intelligence_pipeline.py`)

**Status:** Fully implemented, tested
**Lines:** ~600
**Purpose:** Orchestrates the full intelligence generation flow

**Activation:**
```python
# In search_to_state_ui.py or shelfguard_app.py
from src.intelligence_pipeline import IntelligencePipeline

pipeline = IntelligencePipeline(supabase, openai_api_key, enable_data_accumulation=True)
results = pipeline.generate_portfolio_intelligence(portfolio_asins, market_data, category_context)
```

**Dependencies:**
- `utils/ai_engine_v2.py` - Trigger-aware AI layer
- `utils/ai_engine_v3.py` - Network-aware AI layer
- `src/network_intelligence.py` - Network query layer

---

## Trigger Detection (`src/trigger_detection.py`)

**Status:** Fully implemented
**Lines:** ~500
**Purpose:** Detects actionable trigger events from historical data

**Trigger Types (6):**
1. **Inventory Triggers** - Stock level alerts, reorder signals
2. **Price War Triggers** - Competitive price drops, margin compression
3. **Review Velocity Triggers** - Review bombing, viral reviews
4. **BuyBox Triggers** - BuyBox loss, hijacker detection
5. **Rank Triggers** - BSR spikes/drops, seasonal patterns
6. **Competitor Triggers** - New entrants, competitor OOS

**Activation:**
```python
from src.trigger_detection import TriggerDetector

detector = TriggerDetector()
triggers = detector.detect_triggers(historical_df, current_metrics)
```

---

## Network Intelligence Query Layer (`src/network_intelligence.py`)

**Status:** Writes working, reads NOW ACTIVATED (Fix 1.2)
**Lines:** ~400
**Purpose:** Provides access to accumulated network intelligence

**Now Active Features:**
- `get_category_benchmarks()` - Called in dashboard (Fix 1.2)
- Category median price/BSR/reviews comparison

**Still Staged:**
- `get_competitive_position()` - Full percentile analysis
- `get_brand_intelligence()` - Brand-level aggregates
- `get_historical_pattern()` - Pattern matching ("we've seen this before")
- `get_similar_products()` - Find comparable ASINs

**Activation (for full features):**
```python
from src.network_intelligence import NetworkIntelligence

network = NetworkIntelligence(supabase)
position = network.get_competitive_position(asin, category_id)
brand_intel = network.get_brand_intelligence(brand, category_root)
patterns = network.get_historical_pattern('price_war', category_root)
```

---

## Product Status Model (`src/models/product_status.py`)

**Status:** Fully implemented
**Lines:** ~200
**Purpose:** 13-state product taxonomy with priority scoring

**States:**
- `CRITICAL_DEFEND` (Priority: 100) - Under attack, immediate action needed
- `CRITICAL_OPPORTUNITY` (Priority: 100) - Time-sensitive opportunity
- `HIGH_RISK` (Priority: 75) - Elevated risk, monitor closely
- `GROWING` (Priority: 60) - Positive trajectory
- `STABLE` (Priority: 40) - Healthy, maintain
- ... and 8 more states

**Activation:**
```python
from src.models.product_status import ProductStatus, classify_product_status

status = classify_product_status(metrics, triggers, network_context)
```

---

## Trigger Event Model (`src/models/trigger_event.py`)

**Status:** Fully implemented
**Lines:** ~100
**Purpose:** Dataclass for trigger events with metadata

```python
@dataclass
class TriggerEvent:
    trigger_type: str
    severity: str  # 'critical', 'warning', 'info'
    message: str
    detected_at: datetime
    metrics: Dict[str, Any]
```

---

## Unified Intelligence Output (`src/models/unified_intelligence.py`)

**Status:** Fully implemented
**Lines:** ~150
**Purpose:** Single output object combining all intelligence layers

```python
@dataclass
class UnifiedIntelligence:
    asin: str
    product_status: ProductStatus
    triggers: List[TriggerEvent]
    network_context: Dict[str, Any]
    strategic_brief: StrategicBrief
    recommended_actions: List[str]
    confidence_score: float
```

---

## AI Engine v2 (`utils/ai_engine_v2.py`)

**Status:** Staged
**Lines:** ~400
**Purpose:** Adds trigger detection awareness to strategic classification

**Enhancement over v1:**
- Incorporates trigger events into LLM prompt
- Adjusts confidence based on trigger severity
- Generates trigger-specific recommendations

**Activation:**
```python
from utils.ai_engine_v2 import TriggerAwareTriangulator

triangulator = TriggerAwareTriangulator(openai_api_key)
brief = triangulator.classify_with_triggers(product_data, triggers)
```

---

## AI Engine v3 (`utils/ai_engine_v3.py`)

**Status:** Staged
**Lines:** ~400
**Purpose:** Adds network intelligence context to classification

**Enhancement over v2:**
- Incorporates category benchmarks into analysis
- Adds competitive position context
- Generates network-aware recommendations

**Activation:**
```python
from utils.ai_engine_v3 import NetworkAwareTriangulator

triangulator = NetworkAwareTriangulator(openai_api_key, supabase)
brief = triangulator.classify_with_network(product_data, triggers, category_id)
```

---

## Tables Supporting Staged Features

These Supabase tables are populated but primarily used by staged features:

| Table | Status | Used By |
|-------|--------|---------|
| `category_intelligence` | **NOW READ** (Fix 1.2) | Dashboard, network_intelligence.py |
| `brand_intelligence` | Written only | network_intelligence.py (staged) |
| `market_patterns` | Written only | network_intelligence.py (staged) |
| `strategic_insights` | Written only | intelligence_pipeline.py (staged) |
| `trigger_events` | Written only | intelligence_pipeline.py (staged) |

---

## Activation Roadmap

### Phase 1: Network Context (DONE - Fix 1.2)
- [x] Read category_intelligence in dashboard
- [x] Display competitive position metrics

### Phase 2: Trigger Detection
- [ ] Activate TriggerDetector in dashboard
- [ ] Display trigger alerts in Command Center
- [ ] Add trigger-based notifications

### Phase 3: Full Intelligence Pipeline
- [ ] Activate IntelligencePipeline on project creation
- [ ] Generate UnifiedIntelligence for each ASIN
- [ ] Display strategic_insights in dashboard

### Phase 4: AI Engine v3
- [ ] Switch from ai_engine.py to ai_engine_v3.py
- [ ] Incorporate full network context in classifications
- [ ] Enable pattern-based recommendations

---

**Document Generated:** 2026-01-19
**Total Staged Files:** 8 modules
**Activation Required:** Set flags or call functions as shown above
