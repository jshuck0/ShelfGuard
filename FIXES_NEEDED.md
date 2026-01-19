# Command Center Rendering Issues - Fixes Needed

## Issues Identified

### 1. HTML/CSS Showing in Priority Cards
**Problem:** Raw HTML/CSS is displaying instead of rendering properly
**Location:** `apps/shelfguard_app.py` lines 990-1010
**Root Cause:** The markdown has an issue with the HTML structure or the `unsafe_allow_html` parameter isn't working

### 2. Confidence Scores All Showing as 1%
**Problem:** All confidence scores display as 1% instead of 60-95%
**Location:** Column config at line 1092-1098
**Root Cause:** The analyzer is using fallback logic instead of LLM due to async context detection (line 772 in `ai_engine.py`)

### 3. LLM Not Being Called
**Problem:** Fallback logic is being used even when `use_llm=True`
**Location:** `utils/ai_engine.py` lines 770-773 and 688-691
**Root Cause:** Streamlit runs in an async context, so `asyncio.get_running_loop()` succeeds and triggers fallback

### 4. Single Emoji Column
**Problem:** There's just a column with an emoji instead of proper header
**Location:** Line 1099 - `"Src": st.column_config.TextColumn("", width="small")`
**Root Cause:** Empty string as column name

## Fixes Required

### Fix 1: Update analyze() method to work in Streamlit async context
**File:** `utils/ai_engine.py`
**Change:** Use `nest_asyncio` or create new event loop properly

### Fix 2: Fix HTML rendering in priority cards
**File:** `apps/shelfguard_app.py` 
**Change:** Check for syntax errors in the f-string HTML

### Fix 3: Add proper column header for Source
**File:** `apps/shelfguard_app.py` line 1099
**Change:** `"Src": st.column_config.TextColumn("Source", width="small")`

### Fix 4: Debug confidence score values
**File:** `apps/shelfguard_app.py`
**Add:** Debug print to see actual confidence values being returned
