# Strategic Governor - Example Outputs

## Test Product: "Jif Extra Crunchy Peanut Butter, 28 Oz"

### Product Metrics (Constant Across All Modes)

```json
{
  "asin": "B00WG41HSF",
  "title": "Jif Extra Crunchy Peanut Butter, 28 Oz",
  "filled_price": 6.99,
  "sales_rank_filled": 1250,
  "net_margin": 0.06,  // 6% margin
  "amazon_bb_share": 0.75,
  "velocity_decay": 0.88,  // Improving velocity (rank accelerating)
  "new_offer_count": 12,
  "rank_delta_90d_pct": -12.0,  // Rank improved 12% over 90 days
  "price_gap": 0.05,
  "weekly_sales_filled": 4200
}
```

---

## Mode 1: 💰 Profit Maximization

### AI Classification

```json
{
  "strategic_state": "DISTRESS",
  "confidence": 0.78,
  "reasoning": "Margin at 6% is below acceptable threshold despite rank gains",
  "recommended_action": "Cut all non-essential ad spend. Raise price to $7.49 (+7%)"
}
```

### Dashboard Display

**Priority Card:**
```
#1 PRIORITY: 🚨 DISTRESS (6% CONF)
$4.6K

Recoverable

"Margin at 6% is below acceptable threshold despite rank gains"

Ad: ⏸️ PAUSE & INVESTIGATE
Ecom: 🔍 FIX ROOT CAUSE

[Signals: Margin LOW (6%), Velocity ACCELERATING (0.88x)]
```

### Why This Classification?

In **Profit Mode**, the AI:
- Heavily penalizes the 6% margin (below the 10% threshold for Profit mode)
- Classifies as DISTRESS even though velocity is improving
- Recommends immediate margin recovery actions (price increase, cut spend)
- Treats growth-at-low-margin as unsustainable

---

## Mode 2: ⚖️ Balanced Defense (Default)

### AI Classification

```json
{
  "strategic_state": "TRENCH_WAR",
  "confidence": 0.72,
  "reasoning": "Competitive pressure with 12 sellers. Rank improving but margin compressed",
  "recommended_action": "Monitor margin closely. Maintain defensive position"
}
```

### Dashboard Display

**Priority Card:**
```
#2 PRIORITY: ⚔️ TRENCH WAR (72% CONF)
$4.6K

Defensible

"Competitive pressure with 12 sellers. Rank improving but margin compressed"

Ad: 🎯 DEFENSIVE KEYWORDS
Ecom: 🎫 MATCH COMPETITOR

[Signals: Competition HIGH (12 sellers), Velocity ACCELERATING (0.88x), Margin MODERATE (6%)]
```

### Why This Classification?

In **Balanced Mode**, the AI:
- Sees the 12 competitors and flags competitive pressure
- Acknowledges the improving velocity as a positive signal
- Notes margin is low but not critical
- Classifies as TRENCH_WAR (competitive battle, not crisis)
- Recommends defensive actions (maintain position)

---

## Mode 3: 🚀 Aggressive Growth

### AI Classification

```json
{
  "strategic_state": "HARVEST",
  "confidence": 0.68,
  "reasoning": "Rank improving 12% in 90d. Margin acceptable for growth phase",
  "recommended_action": "Scale ad spend 20%. Strong momentum - invest to capture share"
}
```

### Dashboard Display

**Priority Card:**
```
#5 PRIORITY: 🌾 HARVEST (68% CONF)
$4.6K

Profitable

"Rank improving 12% in 90d. Margin acceptable for growth phase"

Ad: 📉 REDUCE SPEND → 📈 SCALE SPEND
Ecom: 📈 RAISE PRICE → ✅ MAINTAIN

[Signals: Velocity ACCELERATING (0.88x), Rank IMPROVING (-12%), Competition MODERATE (12)]
```

### Why This Classification?

In **Growth Mode**, the AI:
- **Forgives the 6% margin** because rank is improving significantly
- Interprets velocity acceleration as a sign of winning market share
- Sees the 12 competitors as "beatable" given current momentum
- Classifies as HARVEST (not DISTRESS) because growth trajectory is positive
- Recommends **scaling spend** to fuel momentum (opposite of Profit mode)

---

## Side-by-Side Comparison

| Aspect | Profit Mode 💰 | Balanced Mode ⚖️ | Growth Mode 🚀 |
|--------|----------------|------------------|----------------|
| **State** | DISTRESS 🚨 | TRENCH_WAR ⚔️ | HARVEST 🌾 |
| **Confidence** | 78% | 72% | 68% |
| **Primary Focus** | Low margin (6%) | Competition (12 sellers) | Velocity gains (-12%) |
| **Action** | Cut spend, raise price | Defend position | Scale ads 20% |
| **Tone** | "Critical" | "Monitor closely" | "Invest to capture" |
| **Priority** | #1 (urgent) | #2 (important) | #5 (opportunity) |

---

## Key Insights

### The Same Data, Three Different Stories

1. **Profit Mode** sees a product bleeding margin and demands immediate correction
2. **Balanced Mode** sees a competitive situation requiring defensive tactics
3. **Growth Mode** sees a winning product that deserves more investment

### None of These Are "Wrong"

The classification depends entirely on **your current business objective**:

- If you're focused on quarterly profitability → Profit Mode is right
- If you're in steady-state operations → Balanced Mode is right
- If you're building market position → Growth Mode is right

### The Magic: One Click, Instant Reframe

Without the Strategic Governor, you'd need to:
1. Set margin thresholds manually for each product
2. Explain your growth vs. profit priorities
3. Override AI recommendations one by one

With the Strategic Governor:
1. Click "Aggressive Growth"
2. **Every product instantly re-evaluates**
3. Dashboard shows growth-optimized actions

---

## Testing This Yourself

### Step 1: Load Your Dashboard

```bash
streamlit run apps/shelfguard_app.py
```

### Step 2: Observe Default (Balanced) State

- Navigate to Command Center
- Note the top 3 priority cards
- Note their states and recommendations

### Step 3: Switch to Profit Mode

- Click sidebar: **💰 Profit Maximization**
- Watch the dashboard re-render
- Observe:
  - More products classified as DISTRESS/TERMINAL
  - More "Cut spend" and "Raise price" recommendations
  - Priority order may change (margin-critical products rise)

### Step 4: Switch to Growth Mode

- Click sidebar: **🚀 Aggressive Growth**
- Watch the dashboard re-render
- Observe:
  - Fewer products classified as TERMINAL
  - More "Scale ads" and "Expand keywords" recommendations
  - Products with improving velocity get promoted

---

## Real Portfolio Example (Hypothetical)

### Your Portfolio: 20 Products

**Balanced Mode Distribution:**
- FORTRESS: 3 products
- HARVEST: 8 products
- TRENCH_WAR: 6 products
- DISTRESS: 2 products
- TERMINAL: 1 product

**Switch to Profit Mode:**
- FORTRESS: 2 products (stricter margin requirements)
- HARVEST: 6 products
- TRENCH_WAR: 5 products
- DISTRESS: **5 products** (was 2 - margin compression flagged)
- TERMINAL: **2 products** (was 1 - low-margin growth classified as terminal)

**Switch to Growth Mode:**
- FORTRESS: 3 products
- HARVEST: 10 products (was 8 - forgiving on margin if velocity good)
- TRENCH_WAR: 6 products
- DISTRESS: **1 product** (was 2 - only true failures flagged)
- TERMINAL: **0 products** (was 1 - margin forgiven for growth)

---

## Conclusion

The Strategic Governor is a **context injection system** that makes the AI feel intelligent and adaptive. It's the difference between:

**Bad AI:** "This product has a 6% margin. What should I do?"

**Good AI:** "Given your focus on growth this quarter, this product's 6% margin is acceptable because rank is improving 12%. Invest to capture momentum."

One selector. Entire system reframes. That's the magic. ✨
