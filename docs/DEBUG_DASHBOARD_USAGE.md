# LLM Engine Debug Dashboard - Usage Guide

## 🚀 Quick Start

### Run the Debug Dashboard

```bash
cd C:\Users\jshuc\OneDrive\Desktop\ShelfGuard
streamlit run apps/debug_llm_engine.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🔬 Test Areas

### 1. Data Healer Test
**What it tests:** Gap filling and interpolation quality

**What you'll see:**
- Before/After comparison of data with gaps
- Healing time and success rate
- Validation results
- Data quality metrics

**Expected Results:**
- ✅ All gaps filled (0 remaining)
- ✅ Healing time < 0.5s for 10 products
- ✅ Validation: PASSED

### 2. Single Product Test
**What it tests:** LLM classifier vs deterministic fallback on one product

**Test Scenarios Available:**
- **FORTRESS** - Dominant position, low competition
- **HARVEST** - Cash cow, maximize margin
- **TRENCH_WAR** - Competitive attack, defend share
- **DISTRESS** - Margin compression, needs intervention
- **TERMINAL** - Exit required, liquidate

**What you'll see:**
- Side-by-side comparison: LLM vs Fallback
- Strategic state classification
- Confidence scores
- Reasoning and recommended actions
- Response times
- Signal detection

**Expected Results:**
- ✅ LLM confidence: 85-95%
- ✅ Fallback confidence: 60-70%
- ✅ LLM reasoning: Detailed and nuanced
- ✅ Fallback reasoning: Generic rules-based
- ✅ LLM response time: 1-3 seconds
- ✅ Fallback response time: <0.1 seconds

### 3. Competitive Intelligence Test
**What it tests:** Signal extraction for competitive analysis

**What you'll see:**
- Competitive metrics for each scenario
- Seller counts and trends
- Buy Box ownership percentages
- Margin and rank trends
- Expected state classifications

**Expected Results:**
- ✅ Clear differentiation between states based on competitive signals
- ✅ TRENCH_WAR shows high seller count + declining Buy Box
- ✅ FORTRESS shows low seller count + high Buy Box
- ✅ DISTRESS shows margin compression + competitive pressure

### 4. Batch Performance Test
**What it tests:** Parallel processing of multiple products

**What you'll see:**
- Total processing time for 10 products
- Per-product average time
- State distribution across portfolio
- Confidence statistics
- Success rate

**Expected Results:**
- ✅ Total time: <3 seconds for 10 products
- ✅ Success rate: 100%
- ✅ Average confidence: >80%
- ✅ High confidence products (>85%): >70%
- ✅ State distribution: Varied across FORTRESS, HARVEST, TRENCH_WAR, etc.

### 5. All Tests
Runs all the above tests sequentially.

## 📊 What to Share with Me

### For Each Test Run:

1. **Screenshot** the main dashboard showing:
   - Test mode selected
   - All metrics displayed
   - Any error messages

2. **Copy and paste** the following information:

#### Data Healer Test Results:
```
Gaps Before: [NUMBER]
Gaps After: [NUMBER]
Healing Time: [NUMBER]s
Validation: [PASSED/FAILED]
```

#### Single Product LLM Test (pick one scenario):
```
Scenario: [fortress/harvest/trench_war/distress/terminal]

LLM Mode:
- State: [STATE]
- Confidence: [XX]%
- Response Time: [X.XX]s
- Reasoning: [COPY REASONING TEXT]
- Action: [COPY ACTION TEXT]

Fallback Mode:
- State: [STATE]
- Confidence: [XX]%
- Response Time: [X.XX]s

Comparison:
- States Match: [YES/NO]
- Confidence Match: [YES/NO]
```

#### Batch Performance Test Results:
```
Total Time: [X.XX]s
Per Product: [X.XXX]s
Products Classified: [NUMBER]
Success Rate: [XX]%
Average Confidence: [XX]%
High Confidence Products: [NUMBER]/[TOTAL]

State Distribution:
- FORTRESS: [X]
- HARVEST: [X]
- TRENCH_WAR: [X]
- DISTRESS: [X]
- TERMINAL: [X]
```

3. **Export Test Results** (use button in sidebar)
   - Download the JSON file
   - Attach to your message

4. **Console Output** (if any errors)
   - Check your terminal where you ran `streamlit run`
   - Copy any error messages or warnings

## 🐛 Troubleshooting

### Issue: "AI Engine not available"
**Solution:** Ensure OpenAI API key is set in `.streamlit/secrets.toml`:
```toml
[openai]
api_key = "sk-..."
```

### Issue: "Module not found: utils.ai_engine"
**Solution:** Make sure you're running from the ShelfGuard root directory:
```bash
cd C:\Users\jshuc\OneDrive\Desktop\ShelfGuard
```

### Issue: LLM test shows "LLM Error"
**Solutions:**
1. Check OpenAI API key is valid
2. Check internet connection
3. Check OpenAI API status: https://status.openai.com/
4. Try running with "Enable LLM Mode" unchecked (tests fallback only)

### Issue: Slow performance
**Expected behavior:**
- First LLM call may take 2-5 seconds (cold start)
- Subsequent calls should be 1-2 seconds
- Fallback calls should be <0.1 seconds

### Issue: UnicodeEncodeError in console
**Solution:** This is a Windows console encoding issue (emojis). The dashboard should still work fine. The error is cosmetic and only affects terminal output, not the Streamlit UI.

## 🎯 What Good Results Look Like

### Perfect Run Example:

```
✅ Data Healer Test
   - Gaps Before: 10
   - Gaps After: 0
   - Time: 0.15s
   - Validation: PASSED

✅ Single Product Test (TRENCH_WAR scenario)
   LLM:
   - State: TRENCH_WAR
   - Confidence: 94%
   - Time: 2.1s
   - Reasoning: "Significant competitive attack detected. +7 new 
     sellers in 30 days, Buy Box share dropped from 85% → 62%..."
   - Action: "Increase ad spend 30%. Do NOT lower price further."
   
   Fallback:
   - State: TRENCH_WAR
   - Confidence: 70%
   - Time: 0.05s
   
   ✓ States match!

✅ Batch Performance Test (10 products)
   - Total Time: 2.8s
   - Success Rate: 100%
   - Average Confidence: 89%
   - High Confidence: 8/10 (80%)
   - State Distribution: Balanced across all states
```

### Red Flags to Watch For:

❌ Gaps After Healing > 0  
❌ LLM Confidence < 75% consistently  
❌ LLM and Fallback states disagree often (>30% mismatch rate)  
❌ Batch time > 5 seconds for 10 products  
❌ Success rate < 90%  
❌ LLM errors or timeouts  

## 💡 Tips for Testing

1. **Test each scenario individually** first (fortress, harvest, etc.)
   - This helps isolate issues

2. **Run "All Tests" last** for a comprehensive report
   - This gives you the full system picture

3. **Test with LLM enabled first**, then disabled
   - Compare LLM vs Fallback performance

4. **Take screenshots** of interesting results
   - Especially where LLM gives detailed reasoning

5. **Note any patterns** in misclassifications
   - E.g., "LLM classified as DISTRESS but should be TRENCH_WAR"

## 📤 Sending Results to Me

Once you've run the tests, please share:

1. Screenshots of the dashboard (one per test type)
2. The text results formatted as shown above
3. The exported JSON file (use "Export Test Results" button)
4. Any observations or questions you have
5. Console output if there are any errors

Example message format:
```
I ran all the debug tests. Here are the results:

[SCREENSHOT]

Data Healer: PASSED (0 gaps, 0.15s)
Single Product (TRENCH_WAR): LLM=94%, Fallback=70%, States Match ✓
Batch Performance: 2.8s for 10 products, 89% avg confidence

Observations:
- LLM gives much more detailed reasoning than fallback
- All tests passed validation
- One question: Why does fortress scenario show...

[ATTACH JSON FILE]
```

## 🔄 Next Steps After Testing

Based on your results, we'll:
1. Validate that the LLM is working correctly
2. Fine-tune confidence thresholds if needed
3. Adjust competitive signal weights
4. Optimize prompts for better reasoning
5. Integrate into your main ShelfGuard dashboard

---

**Ready to test?** Run:
```bash
streamlit run apps/debug_llm_engine.py
```

Then share your results!
