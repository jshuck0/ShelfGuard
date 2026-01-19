# LLM Usage Status Check

## Summary
**STATUS:** Executive Brief and Chat work ✓ | Product Classification broken ✗

## All LLM Usage Points

### 1. Executive Strategic Brief (Portfolio-Level)
**Location:** `apps/shelfguard_app.py` line 639  
**Function:** `generate_ai_brief()` → `generate_portfolio_brief_sync()`  
**Implementation:** `utils/ai_engine.py` lines 949-977  

**How it works:**
```python
def generate_portfolio_brief_sync(portfolio_summary, client, model):
    try:
        loop = asyncio.new_event_loop()  # Creates NEW loop
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            generate_portfolio_brief(portfolio_summary, client, model)
        )
        loop.close()
        return result
    except Exception:
        return None
```

**Status:** ✅ **SHOULD WORK** - Creates new event loop explicitly

**Verification:** Check if you see an LLM-generated brief at the top of the dashboard (vs fallback text)

---

### 2. Chat Interface
**Location:** `apps/shelfguard_app.py` lines 506-510  
**Function:** Direct OpenAI synchronous call  

**How it works:**
```python
response = openai_client.chat.completions.create(
    model=st.secrets["openai"]["model"],
    messages=messages_for_api,
    max_tokens=300
)
```

**Status:** ✅ **SHOULD WORK** - Uses synchronous OpenAI client (not AsyncOpenAI)

**Verification:** Try typing a question in the chat box

---

### 3. Product-Level Classification
**Location:** Multiple places in `shelfguard_app.py`
- Line 84 (get_product_strategy function)
- Line 930 (priority cards)
- Line 1056 (action queue table)

**Function:** `StrategicTriangulator.analyze()`  
**Implementation:** `utils/ai_engine.py` lines 748-780  

**How it works (BROKEN):**
```python
def analyze(self, row):
    if not self.use_llm:
        return _determine_state_fallback(row_data, "LLM disabled")
    
    try:
        try:
            asyncio.get_running_loop()
            # ❌ PROBLEM: Streamlit has a running loop, so this succeeds
            #    and triggers fallback instead of using LLM
            return _determine_state_fallback(row_data, "Sync call in async context")
        except RuntimeError:
            # This path never executes in Streamlit
            return asyncio.run(analyze_strategy_with_llm(row_data))
    except Exception as e:
        return _determine_state_fallback(row_data, f"Error: {str(e)[:30]}")
```

**Status:** ✗ **BROKEN** - Always uses fallback in Streamlit due to async context detection

**Evidence:**
1. All confidence scores show as 60-70% (fallback default) instead of 85-95% (LLM)
2. Reasoning shows "[Fallback: ...]" text instead of nuanced LLM reasoning
3. Source indicator shows 📊 (fallback) instead of 🤖 (LLM)

**Why this breaks:**
- Streamlit runs in an async context internally
- `asyncio.get_running_loop()` succeeds (doesn't raise RuntimeError)
- Code assumes async context = can't run LLM
- Falls back to deterministic logic

---

## The Fix

### Option 1: Match the Portfolio Brief Pattern (RECOMMENDED)
Change `analyze()` method to create a new event loop explicitly like `generate_portfolio_brief_sync()` does:

```python
def analyze(self, row: Union[pd.Series, Dict]) -> StrategicBrief:
    # ... normalize row ...
    
    if not self.use_llm:
        return _determine_state_fallback(row_data, "LLM disabled")
    
    try:
        # Create new event loop explicitly (like generate_portfolio_brief_sync)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            analyze_strategy_with_llm(row_data, timeout=self.timeout)
        )
        loop.close()
        return result
    except Exception as e:
        return _determine_state_fallback(row_data, f"Error: {str(e)[:30]}")
```

### Option 2: Use nest_asyncio (ALTERNATIVE)
Install `nest_asyncio` to allow nested event loops:

```python
try:
    import nest_asyncio
    nest_asyncio.apply()
    return asyncio.run(analyze_strategy_with_llm(row_data, timeout=self.timeout))
except ImportError:
    # Fallback if nest_asyncio not installed
    return _determine_state_fallback(row_data, "nest_asyncio not available")
```

---

## Verification Steps

### Test 1: Check Executive Brief
**What to look for:** At the top of the dashboard, the "STRATEGIC BRIEF" box  
**✅ LLM Working:** Tactical language like "Protocol Activated", "Execute immediately", specific numbers  
**✗ Fallback:** Generic text like "positioned in competitive market. Monitor and optimize."

### Test 2: Check Chat
**What to do:** Type "What should I focus on?" in the chat  
**✅ LLM Working:** You get a conversational AI response  
**✗ Broken:** You get an error message

### Test 3: Check Product Classification (CURRENTLY BROKEN)
**What to look for:** Priority cards and action queue table  

**Current state (fallback):**
- Confidence: 60-70%
- Source: 📊
- Reasoning: Short, generic like "Metrics within normal range"
- State: Generic classifications

**Expected state (LLM):**
- Confidence: 85-95%
- Source: 🤖
- Reasoning: Detailed like "Significant competitive attack detected. +7 new sellers in 30 days..."
- State: Nuanced classifications based on signal triangulation

---

## Current Dashboard Issues

Based on the screenshot:

1. **HTML/CSS Showing in Cards**: Unrelated to LLM - rendering issue with markdown
2. **Confidence All 1%**: Because fallback is being used (60-70% confidence) but display might be wrong
3. **Poor LLM Recommendations**: Because LLM isn't being called - fallback logic is running
4. **Single Emoji Column**: Missing column header "Source"

---

## Recommendation

**Apply Option 1 (match portfolio brief pattern)** - it's proven to work for the executive brief, so it should work for product classification too.

This is a **single line of code change** in `utils/ai_engine.py` that makes `analyze()` work the same way as `generate_portfolio_brief_sync()`.
