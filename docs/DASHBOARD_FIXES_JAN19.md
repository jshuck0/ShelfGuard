# Dashboard Fixes - January 19, 2026

## Issues Addressed

### 1. Executive Brief - Military Language Removal
**Problem:** Executive brief used too much military/tactical language ("threat detected", "defense perimeter", "protocol activated")

**Fix:** Updated system prompt in `utils/ai_engine.py` (lines 915-931)
- Removed military terminology
- Changed to clear, direct business language
- Updated guidelines to explicitly avoid tactical jargon
- Examples: "defense perimeter" → "portfolio status", "threat detection" → "key issue"

**Files Changed:**
- `utils/ai_engine.py` - Updated `generate_portfolio_brief` system prompt

---

### 2. Priority Card Text Truncation
**Problem:** LLM reasoning text in priority cards was getting cut off at 100 characters

**Fix:** Reduced truncation limit in `apps/shelfguard_app.py` (line 984)
- Changed from 100 characters to 60 characters
- Ensures text fits within card boundaries
- Added ellipsis for longer text

**Files Changed:**
- `apps/shelfguard_app.py` - Line 984, reasoning preview truncation

---

### 3. Action Queue Issues
**Multiple Problems:**
- LLM recommendations were inconsistent/not insightful
- Confidence scores all showing as 1% or 0%
- Floating emoji column at end of table

**Fixes:**

#### 3a. Improved LLM Recommendation Prompt
Updated product classification prompt in `utils/ai_engine.py` (lines 198-210)
- Added explicit instructions for short, specific recommendations
- Required reasoning under 80 characters
- Required specific, measurable actions
- Added business language requirement

#### 3b. Fixed Confidence Score Display
**Root Cause:** Legacy fallback was missing `confidence_score` field, causing defaults to 0

**Fix in `apps/shelfguard_app.py`:**
- Added `confidence_score: 0.5` to legacy fallback (line 162)
- Added `confidence_score: 0.3` to error fallback (line 171)
- Added missing fields (`strategic_state`, `recommended_plan`) to both fallbacks

#### 3c. Removed Floating Source Column
**Problem:** Source indicator was in separate "Src" column with just emoji

**Fix in `apps/shelfguard_app.py`:**
- Combined source emoji with Action text (lines 1073-1086)
- Removed "Src" column from column configuration (line 1102)
- Source now shows as prefix in Action column: "🤖 Reduce ad spend 20%"

#### 3d. Improved Action Display
- Increased action display length from 40 to 80 characters (line 1077)
- Show first sentence or 80 chars to provide full context
- Renamed column from "LLM Recommendation" to "AI Recommendation"

**Files Changed:**
- `utils/ai_engine.py` - Updated product classification prompt
- `apps/shelfguard_app.py` - Lines 151-174 (fallback fixes), 1073-1102 (action queue fixes)

---

## Testing Checklist

- [ ] Executive brief uses clear business language (no military terms)
- [ ] Priority cards show reasoning without cutoff
- [ ] Confidence scores display correctly (not all 0% or 1%)
- [ ] AI recommendations are specific and actionable
- [ ] Action queue has no floating columns
- [ ] Source emoji appears in Action column (🤖 = LLM, 📊 = fallback, ⚙️ = legacy)

---

## Restart Instructions

```bash
streamlit run apps/shelfguard_app.py
```

Navigate to Command Center and verify all three issues are resolved.
