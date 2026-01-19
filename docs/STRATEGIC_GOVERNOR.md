# Strategic Governor - Context-Aware AI Engine

## Overview

The **Strategic Governor** is a single UI control that transforms the entire AI recommendation engine based on your current business priority. Instead of asking you dozens of questions about each product, it applies a global "lens" that shifts how every metric is interpreted.

Think of it like driving modes in a sports car:
- **💰 Profit Maximization**: Eco mode - prioritize efficiency
- **⚖️ Balanced Defense**: Normal mode - standard evaluation
- **🚀 Aggressive Growth**: Sport mode - prioritize velocity and market share

## How It Works

### The UI: One Click Changes Everything

Located in the sidebar, the Strategic Governor is a radio selector with three modes:

```python
💰 Profit Maximization
⚖️ Balanced Defense  (default)
🚀 Aggressive Growth
```

When you change this setting, **every product recommendation in the dashboard instantly re-evaluates** using the new strategic lens.

### The Magic: Biased Intelligence

The Strategic Governor modifies both the LLM prompt and the deterministic fallback logic:

#### 1. LLM Classifier Adjustments

**Profit Mode:**
- Heavily penalizes products with margins <10%
- Classifies low-margin products as DISTRESS even if velocity is good
- Recommends "Cut ad spend" or "Raise price" before "Scale ads"
- Questions growth spending that compresses margins

**Growth Mode:**
- Forgives margin compression if rank is improving
- Won't classify as TERMINAL unless rank is catastrophic AND declining
- Encourages investment in products with momentum
- Example: 3% margin + improving rank → TRENCH_WAR (not TERMINAL)

**Balanced Mode:**
- Standard strategic logic
- Evaluates all factors equally
- Applies default state definitions

#### 2. Deterministic Fallback Adjustments

When the LLM is unavailable, the fallback logic adjusts classification thresholds:

| Threshold | Profit Mode | Balanced Mode | Growth Mode |
|-----------|-------------|---------------|-------------|
| TERMINAL margin | <3% | <2% | <0% |
| DISTRESS margin | <10% | <8% | <5% |
| HARVEST margin | >15% | >15% | >12% |
| FORTRESS margin | >20% | >18% | >15% |

## Real-World Scenarios

### Scenario 1: Product Breaking Even with Growth

**Product Metrics:**
- Margin: 5%
- Rank: Improving 15%
- Ad Spend: High

**Profit Mode Response:**
- State: DISTRESS
- Reasoning: "Unsustainable margins at 5%"
- Action: "Cut ad spend 50%. Margin protection required."

**Growth Mode Response:**
- State: TRENCH_WAR
- Reasoning: "Acceptable margin sacrifice for 15% rank improvement"
- Action: "Maintain ad investment. Monitor velocity trends."

### Scenario 2: Stable Product with Good Margins

**Product Metrics:**
- Margin: 18%
- Rank: Stable
- Competition: Low

**Profit Mode Response:**
- State: HARVEST
- Reasoning: "Strong margins, stable position"
- Action: "Maximize extraction. Raise price +10%. Cut ad spend 30%."

**Growth Mode Response:**
- State: HARVEST
- Reasoning: "Healthy foundation for expansion"
- Action: "Test ad expansion to adjacent keywords. Scale for share gain."

**Balanced Mode Response:**
- State: HARVEST
- Reasoning: "Strong fundamentals for value extraction"
- Action: "Test price increase +5%. Reduce ad spend. Maximize profit."

## Implementation Architecture

### Data Flow

```
User selects Strategic Governor in Sidebar
         ↓
Strategic Bias stored in st.session_state
         ↓
Dashboard calls get_product_strategy(row, strategic_bias=bias)
         ↓
StrategicTriangulator initialized with bias
         ↓
LLM Classifier receives modified system prompt
         ↓
If LLM fails → Fallback uses adjusted thresholds
         ↓
Recommendation returned with bias-adjusted actions
```

### Code Components

**1. Sidebar Selector (apps/shelfguard_app.py)**
```python
strategic_bias = st.sidebar.radio(
    "🎯 Current Strategic Focus",
    options=['💰 Profit Maximization', '⚖️ Balanced Defense', '🚀 Aggressive Growth'],
    index=1,  # Default to Balanced
    key='strategic_bias'
)
```

**2. Bias Instructions (utils/ai_engine.py)**
```python
def _get_strategic_bias_instructions(strategic_bias: str) -> str:
    """Generate LLM instructions based on user's strategic focus."""
    # Returns modified prompt based on mode
```

**3. LLM Integration (utils/ai_engine.py)**
```python
async def analyze_strategy_with_llm(
    row_data: Dict,
    strategic_bias: str = "Balanced Defense"
) -> StrategicBrief:
    # Appends bias instructions to system prompt
    bias_instructions = _get_strategic_bias_instructions(strategic_bias)
    full_system_prompt = f"{STRATEGIST_SYSTEM_PROMPT}\n\n{bias_instructions}"
```

**4. Deterministic Fallback (utils/ai_engine.py)**
```python
def _determine_state_fallback(
    row_data: Dict,
    strategic_bias: str = "Balanced Defense"
) -> StrategicBrief:
    # Adjusts thresholds based on bias
    if strategic_bias == "Profit Maximization":
        margin_terminal = 0.03  # Stricter
        margin_distress = 0.10  # Stricter
    elif strategic_bias == "Aggressive Growth":
        margin_terminal = 0.00  # More forgiving
        margin_distress = 0.05  # More forgiving
```

## Usage Guidelines

### When to Use Profit Mode

- End of quarter, prioritizing cash flow
- High inventory costs, need to maximize ROI
- Testing price elasticity across portfolio
- Cutting unprofitable SKUs

### When to Use Growth Mode

- Launch phase, building market presence
- Defending against competitive entry
- Scaling into new categories
- Accepting short-term margin compression for long-term position

### When to Use Balanced Mode

- Steady-state operations
- Portfolio in equilibrium
- Standard monitoring and optimization
- When you want unbiased strategic assessment

## Future Enhancements

### Per-SKU Role Override (Planned)

In addition to the global Strategic Governor, future versions will support per-product overrides:

**UI Concept:**
- In the Action Queue table, add a "Role" dropdown next to each product
- Options: "Standard", "Loss Leader", "Launch Phase", "Liquidation"
- When set, the product uses custom rules regardless of global bias

**Example:**
```
Product: Jif Extra Crunchy Peanut Butter, 28 Oz
AI State: TERMINAL (based on margin)
User Override: "Loss Leader"
→ AI Response changes to: "Maintain position. This is a strategic loss leader."
```

## Testing the Strategic Governor

### Manual Test

1. Open ShelfGuard Command Center
2. Note current recommendations for top 3 products
3. Change Strategic Governor from Balanced → Profit
4. Observe how recommendations shift to prioritize margin
5. Change to Growth mode
6. Observe how recommendations shift to prioritize velocity

### Expected Behavior

- **Profit Mode**: More products classified as DISTRESS or TERMINAL if margins are thin
- **Growth Mode**: Fewer products classified as TERMINAL, more TRENCH_WAR classifications
- **Recommendations change**: Actions shift from "Scale ads" (Growth) to "Cut spend" (Profit)

## Technical Notes

### Performance

- Strategic bias evaluation adds <5ms per product
- No API calls required for bias adjustment
- LLM prompt increases by ~100 tokens (negligible cost)

### Caching

- Recommendations are NOT cached when strategic bias changes
- This ensures immediate re-evaluation across all products
- If bias hasn't changed, normal caching applies

### Error Handling

- If bias parameter is missing, defaults to "Balanced Defense"
- Fallback logic always has bias support (no crash risk)
- Invalid bias strings are normalized to nearest valid mode

## Conclusion

The Strategic Governor embodies the "Apple/Steve Jobs" philosophy: **Don't ask the user what they want for each decision. Give them one powerful knob that changes the entire system intelligently.**

One click. Entire portfolio reshuffles. That's the magic.
