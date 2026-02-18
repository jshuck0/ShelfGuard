"""
Sherlock Engine: 4-Step LLM Critic Loop

The brain of the AI Analyst. Runs a multi-step reasoning chain:
1. Analyst: Find patterns in the event stream
2. Skeptic: Verify causality, assign confidence
3. Oracle: Predict outcomes, recommend actions
4. Red Team: Roleplay as competitor, find vulnerabilities

This produces the Daily Brief with 3 strategic narratives.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Try to import OpenAI
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

# Try to import Streamlit for secrets
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.analyst.models import (
    StrategicNarrative, 
    DailyBrief, 
    EnrichedEvent,
    NarrativeType,
    ActionUrgency,
)
from src.analyst.event_stream import format_events_for_llm, get_event_summary
from src.analyst.journal import format_journal_context
from src.analyst.world_context import format_world_context_for_llm


# =============================================================================
# LLM CLIENT (reuse patterns from ai_engine.py)
# =============================================================================

def _get_openai_client() -> Optional[AsyncOpenAI]:
    """Get async OpenAI client."""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = None
    
    if STREAMLIT_AVAILABLE:
        try:
            api_key = st.secrets.get("openai", {}).get("OPENAI_API_KEY")
        except Exception:
            pass
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    return AsyncOpenAI(api_key=api_key)


def _get_model_name() -> str:
    """Get model name from secrets or default."""
    if STREAMLIT_AVAILABLE:
        try:
            model = st.secrets.get("openai", {}).get("model")
            if model:
                return model
        except Exception:
            pass
    return "gpt-4o-mini"


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

ANALYST_PROMPT = """You are a creative market analyst examining 90 days of market events for an Amazon seller.

{world_context}

{journal_context}

{event_stream}

=== YOUR TASK ===
Find the 10 most interesting PATTERNS, OPPORTUNITIES, and THREATS in this data.

BE CREATIVE. Brainstorm freely. Include wild ideas - we will filter later.
Look for non-obvious connections. What hidden stories does this data tell?

For each pattern:
1. Describe WHAT happened (specific events, dates, products)
2. Identify CHAIN REACTIONS (did Event A consistently precede Event B?)
3. Quantify the IMPACT (revenue, rank, share)
4. Note which products are YOURS (📍YOUR) vs COMPETITORS (🎯COMP)
5. Suggest a STRATEGIC OPPORTUNITY or THREAT this creates

Consider:
- Competitor OOS events = conquest opportunities
- Price movements that preceded rank changes
- Review surges or drops that drove sales
- Buy Box shifts between sellers
- Seasonal patterns and upcoming events
- Velocity trends (accelerating vs decelerating sales)
- OOS risk indicators
- Seller count changes (competitive pressure)

Output as JSON:
{{
    "patterns": [
        {{
            "title": "Brief catchy title",
            "description": "What happened",
            "chain_reaction": "Event A led to Event B",
            "products_involved": ["ASIN1", "ASIN2"],
            "your_products_affected": true,
            "estimated_impact": "$X or X%",
            "dates": "Jan 15-20",
            "opportunity_or_threat": "What strategic action this suggests"
        }}
    ]
}}
"""

SKEPTIC_PROMPT = """You are a skeptic reviewing an analyst's findings. Your job is to VERIFY or REJECT each pattern.

=== ANALYST'S FINDINGS ===
{analyst_output}

=== YOUR TASK ===
For each pattern, check:

1. TIMING: Did the cause happen BEFORE the effect? (Check dates carefully)
2. ALTERNATIVES: Could something else explain this? (Seasonality? Amazon algo? Coincidence?)
3. EVIDENCE: Is there enough data to support this claim?
4. SIGNIFICANCE: Is this actually important, or just noise?

Be harsh. If a pattern doesn't hold up, DROP IT.

Output as JSON:
{{
    "verified_patterns": [
        {{
            "title": "Pattern title",
            "original_claim": "What analyst said",
            "verification_status": "VERIFIED" | "WEAK" | "REJECTED",
            "confidence": 0.0-1.0,
            "reasoning": "Why you believe or doubt this",
            "alternative_explanation": "What else could explain this",
            "adjusted_impact": "$X or X%"
        }}
    ],
    "rejected_count": 0,
    "overall_data_quality": "HIGH" | "MEDIUM" | "LOW"
}}
"""

EDITOR_PROMPT = """You are the Senior Editor at a strategic intelligence agency. Your job is to KILL bad ideas.

=== PRODUCT IDENTITY ===
Category: {category}
Brand: {brand}
Product Types: {product_types}

=== PORTFOLIO INVENTORY ===
{inventory_list}

=== DRAFT INSIGHTS FROM ANALYST ===
{raw_insights}

=== YOUR TASK ===
Review each draft insight and KILL any that fail these tests:

1. RELEVANCE (Category Fit)
   - Does this strategy physically apply to {category}?
   - "Snack promotion" for Hair Care = KILL
   - "Party-ready hair" for Super Bowl = KEEP (tangential angle is OK)
   - Score 0-10. If < 5, KILL.

2. FEASIBILITY (7-Day Execution)
   - Can this be executed in 7 days with existing inventory?
   - "Launch new product line" = KILL (takes months)
   - "Create virtual bundle of existing SKUs" = KEEP
   - "Increase PPC on existing SKU" = KEEP

3. SPECIFICITY (Actionable)
   - Does it name specific products, prices, competitors?
   - "Optimize your marketing" = KILL (too vague)
   - "Increase PPC on B08XYZ by 40% to capture Native's stockout" = KEEP

4. TONE (Strategic vs Gimmicky)
   - Would a Fortune 500 brand manager take this seriously?
   - "Super Bowl Hair Party!" = KILL (gimmicky)
   - "Capitalize on competitor stockout during high-traffic period" = KEEP

=== OUTPUT ===
Return ONLY the top 3 survivors, ranked by expected impact.

Output as JSON:
{{
    "killed_count": 7,
    "kill_reasons": [
        {{"insight": "Insight title", "reason": "Why killed", "test_failed": "RELEVANCE|FEASIBILITY|SPECIFICITY|TONE"}}
    ],
    "survivors": [
        {{
            "rank": 1,
            "original_title": "Insight title",
            "original_insight": "Full insight description",
            "why_it_survived": "Which tests it passed and why",
            "relevance_score": 8,
            "execution_notes": "Any modifications needed for feasibility",
            "estimated_impact": "$X"
        }}
    ]
}}
"""

ORACLE_PROMPT = """You are a strategic advisor for an Amazon seller. Based on EDITOR-APPROVED insights, predict what will happen and what to do.

{world_context}

=== VERIFIED PATTERNS ===
{verified_patterns}

=== YOUR TASK ===
Create exactly 3 Strategic Narratives, prioritized by impact.

For each narrative:
1. TITLE: Catchy name (e.g., "The Salt & Stone Opportunity")
2. TYPE: CONQUEST (opportunity) | THREAT (defensive) | OPTIMIZATION (efficiency)
3. WHAT HAPPENED: 2-3 sentences summarizing the verified pattern
4. PREDICTION: What will happen in the next 7-14 days if current trends continue
5. EXPECTED IMPACT: Quantify in $ (revenue gain/loss)
6. TRIGGER TO WATCH: What event would change your prediction
7. RECOMMENDED ACTION: Specific, quantified action (e.g., "Increase PPC 40%", not "Optimize marketing")
8. URGENCY: NOW (act in 24hrs) | THIS_WEEK | MONITOR

Be SPECIFIC. Name products and competitors. Quantify everything in $.

Output as JSON:
{{
    "market_summary": "One sentence overview",
    "narratives": [
        {{
            "title": "The X Opportunity",
            "type": "CONQUEST",
            "pattern_summary": "What happened",
            "prediction": "What will happen",
            "expected_impact": 5000,
            "trigger_to_watch": "What to monitor",
            "reversal_risk": "What could go wrong",
            "recommended_action": "Specific action",
            "action_urgency": "NOW",
            "action_rationale": "Why this action",
            "confidence": 0.75
        }}
    ],
    "key_risks": ["Risk 1", "Risk 2"],
    "key_opportunities": ["Opp 1", "Opp 2"]
}}
"""

RED_TEAM_PROMPT = """You are the CEO of {competitor_brand}, the main competitor to {your_brand}.

Look at the market data and find their vulnerabilities.

{event_stream_summary}

=== YOUR TASK ===
Think like an attacker. Where is {your_brand} vulnerable RIGHT NOW?

Consider:
1. Where are they low on inventory?
2. Where is their rating weak?
3. Where are they overpriced?
4. Where have they been losing rank?
5. Where could you launch conquest ads effectively?

If you had $10,000 to spend to STEAL their market share, where would you strike?

Be ruthless. Find the attack vector they haven't considered.

Output as JSON:
{{
    "primary_vulnerability": "Their biggest weakness",
    "attack_vector": "How I would exploit it",
    "budget_allocation": "How I'd spend $10K",
    "expected_damage": "Revenue I could capture",
    "their_blind_spot": "What they're not seeing",
    "defensive_recommendation": "What they should do to protect themselves"
}}
"""


# =============================================================================
# LLM CALL HELPERS
# =============================================================================

async def _call_llm(
    client: AsyncOpenAI,
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """Make an LLM call and return the response text."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strategic market analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return "{}"


def _parse_json_safely(text: str) -> Dict:
    """Parse JSON from LLM response, handling common issues."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        return {}


# =============================================================================
# THE 4-STEP CRITIC LOOP
# =============================================================================

async def run_analyst_step(
    client: AsyncOpenAI,
    events: List[EnrichedEvent],
    world_context: str,
    journal_context: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Step 1: Analyst finds patterns in the event stream."""
    event_stream = format_events_for_llm(events, max_tokens=60000)
    
    prompt = ANALYST_PROMPT.format(
        world_context=world_context,
        journal_context=journal_context,
        event_stream=event_stream,
    )
    
    response = await _call_llm(client, prompt, model=model, max_tokens=2000)
    return _parse_json_safely(response)


async def run_skeptic_step(
    client: AsyncOpenAI,
    analyst_output: Dict,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Step 2: Skeptic verifies causality and assigns confidence."""
    prompt = SKEPTIC_PROMPT.format(
        analyst_output=json.dumps(analyst_output, indent=2)
    )
    
    response = await _call_llm(client, prompt, model=model, max_tokens=2000)
    return _parse_json_safely(response)


def extract_product_identity(events: List[EnrichedEvent]) -> Dict[str, Any]:
    """
    Extract product identity from events for the Editor's context.
    
    Returns:
        Dict with category, brand, product_types, and inventory list
    """
    from src.analyst.models import EventOwner
    
    # Collect portfolio products
    portfolio_events = [e for e in events if e.owner == EventOwner.PORTFOLIO]
    
    # Extract brands
    brands = set()
    for e in portfolio_events:
        if e.brand:
            brands.add(e.brand)
    brand = ", ".join(sorted(brands)[:3]) if brands else "Your Brand"
    
    # Extract product types from semantic tags
    product_types = set()
    for e in portfolio_events:
        for tag in e.tags:
            if tag in ["shampoo", "conditioner", "body-wash", "deodorant", "antiperspirant", 
                       "lotion", "cream", "gel", "spray", "stick"]:
                product_types.add(tag)
    product_types_str = ", ".join(sorted(product_types)) if product_types else "Personal Care"
    
    # Infer category from product types
    if any(t in product_types for t in ["shampoo", "conditioner"]):
        category = "Hair Care"
    elif any(t in product_types for t in ["deodorant", "antiperspirant"]):
        category = "Deodorant & Antiperspirant"
    elif any(t in product_types for t in ["body-wash", "lotion"]):
        category = "Body Care"
    else:
        category = "Personal Care"
    
    # Build inventory list (ASIN + title)
    inventory = []
    seen_asins = set()
    for e in portfolio_events:
        if e.asin not in seen_asins:
            inventory.append(f"{e.asin}: {e.title[:60]}..." if len(e.title) > 60 else f"{e.asin}: {e.title}")
            seen_asins.add(e.asin)
    inventory_str = "\n".join(inventory[:20]) if inventory else "No portfolio products detected"
    
    return {
        "category": category,
        "brand": brand,
        "product_types": product_types_str,
        "inventory_list": inventory_str,
    }


async def run_editor_step(
    client: AsyncOpenAI,
    skeptic_output: Dict,
    product_identity: Dict[str, Any],
    model: str = "gpt-4o-mini",
) -> Dict:
    """
    Step 2.5 (NEW): Editor kills bad ideas before they reach Oracle.
    
    The Editor applies 4 kill-tests:
    1. RELEVANCE - Does it fit the product category?
    2. FEASIBILITY - Can it be executed in 7 days?
    3. SPECIFICITY - Is it actionable with named products?
    4. TONE - Is it strategic, not gimmicky?
    """
    # Format verified patterns from skeptic
    verified = skeptic_output.get("verified_patterns", [])
    raw_insights = json.dumps(verified, indent=2)
    
    prompt = EDITOR_PROMPT.format(
        category=product_identity.get("category", "Personal Care"),
        brand=product_identity.get("brand", "Your Brand"),
        product_types=product_identity.get("product_types", ""),
        inventory_list=product_identity.get("inventory_list", ""),
        raw_insights=raw_insights,
    )
    
    response = await _call_llm(client, prompt, model=model, max_tokens=2000)
    return _parse_json_safely(response)


async def run_oracle_step(
    client: AsyncOpenAI,
    editor_output: Dict,
    world_context: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Step 3: Oracle predicts outcomes and recommends actions based on EDITOR-APPROVED insights."""
    # Use editor survivors, not raw skeptic output
    survivors = editor_output.get("survivors", [])
    verified_patterns = {"verified_patterns": survivors}
    
    prompt = ORACLE_PROMPT.format(
        world_context=world_context,
        verified_patterns=json.dumps(verified_patterns, indent=2),
    )
    
    response = await _call_llm(client, prompt, model=model, max_tokens=2500)
    return _parse_json_safely(response)


async def run_red_team_step(
    client: AsyncOpenAI,
    events: List[EnrichedEvent],
    your_brand: str,
    competitor_brand: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Step 4: Red Team finds vulnerabilities from competitor perspective."""
    # Summarize events for red team
    summary = get_event_summary(events)
    summary_str = json.dumps(summary, indent=2)
    
    prompt = RED_TEAM_PROMPT.format(
        competitor_brand=competitor_brand or "Your Main Competitor",
        your_brand=your_brand or "Your Brand",
        event_stream_summary=summary_str,
    )
    
    response = await _call_llm(client, prompt, model=model, max_tokens=1500)
    return _parse_json_safely(response)


# =============================================================================
# MAIN SHERLOCK FUNCTION
# =============================================================================

async def run_sherlock_analysis(
    events: List[EnrichedEvent],
    project_id: str,
    your_brand: str = "",
    competitor_brand: str = "",
    world_context_dict: Optional[Dict] = None,
    category: str = "default",
    archetype_context: Optional[str] = None,
) -> DailyBrief:
    """
    Run the complete 4-step Sherlock analysis.

    Args:
        events: Sparse event stream from transform_to_event_stream()
        project_id: For journal lookup
        your_brand: Your brand name for Red Team
        competitor_brand: Main competitor for Red Team
        world_context_dict: Optional pre-computed world context
        category: Product category for seasonality
        archetype_context: Optional archetype guidance (from get_archetype_guidance())

    Returns:
        DailyBrief with 3 strategic narratives
    """
    # Get LLM client
    client = _get_openai_client()
    if client is None:
        return _create_fallback_brief(events, project_id)
    
    model = _get_model_name()
    
    # Prepare context
    if world_context_dict is None:
        from src.analyst.world_context import get_world_context
        world_context_dict = get_world_context(category=category)
    
    world_context_str = format_world_context_for_llm(world_context_dict)
    journal_context_str = format_journal_context(project_id)
    
    # Extract product identity for Editor
    product_identity = extract_product_identity(events)
    print(f"📋 Product Identity: {product_identity['category']} | {product_identity['brand']}")
    
    # Run the 5-step loop (Analyst → Skeptic → Editor → Oracle → Red Team)
    print("🔍 Step 1: Analyst finding patterns (10 ideas, high creativity)...")

    # Inject archetype context if available
    analyst_world_context = world_context_str
    if archetype_context:
        analyst_world_context = f"{world_context_str}\n\n{archetype_context}"
        print(f"   (With archetype context)")

    analyst_output = await run_analyst_step(
        client, events, analyst_world_context, journal_context_str, model
    )
    pattern_count = len(analyst_output.get("patterns", []))
    print(f"   Found {pattern_count} patterns")
    
    print("🤔 Step 2: Skeptic verifying causality...")
    skeptic_output = await run_skeptic_step(client, analyst_output, model)
    verified_count = len(skeptic_output.get("verified_patterns", []))
    print(f"   {verified_count} patterns verified")
    
    print("✂️ Step 3: Editor filtering bad ideas (Relevance, Feasibility, Specificity, Tone)...")
    editor_output = await run_editor_step(client, skeptic_output, product_identity, model)
    killed_count = editor_output.get("killed_count", 0)
    survivors = editor_output.get("survivors", [])
    print(f"   Killed {killed_count} ideas, {len(survivors)} survivors")
    
    print("🔮 Step 4: Oracle predicting outcomes from approved insights...")
    oracle_output = await run_oracle_step(
        client, editor_output, world_context_str, model
    )
    
    print("⚔️ Step 5: Red Team finding vulnerabilities...")
    red_team_output = await run_red_team_step(
        client, events, your_brand, competitor_brand, model
    )
    
    # Build the Daily Brief
    brief = _build_daily_brief(
        oracle_output=oracle_output,
        red_team_output=red_team_output,
        editor_output=editor_output,
        product_identity=product_identity,
        world_context=world_context_dict,
        journal_context=journal_context_str,
        events=events,
        project_id=project_id,
    )
    
    return brief


def _build_daily_brief(
    oracle_output: Dict,
    red_team_output: Dict,
    editor_output: Dict,
    product_identity: Dict[str, Any],
    world_context: Dict,
    journal_context: str,
    events: List[EnrichedEvent],
    project_id: str,
) -> DailyBrief:
    """Build DailyBrief from LLM outputs."""
    
    # Parse narratives from Oracle output
    narratives = []
    for n_data in oracle_output.get("narratives", [])[:3]:
        narrative_type = NarrativeType.OPTIMIZATION
        type_str = n_data.get("type", "").upper()
        if type_str == "CONQUEST":
            narrative_type = NarrativeType.CONQUEST
        elif type_str == "THREAT":
            narrative_type = NarrativeType.THREAT
        
        urgency = ActionUrgency.MONITOR
        urgency_str = n_data.get("action_urgency", "").upper()
        if urgency_str == "NOW":
            urgency = ActionUrgency.NOW
        elif urgency_str == "THIS_WEEK":
            urgency = ActionUrgency.THIS_WEEK
        
        narratives.append(StrategicNarrative(
            title=n_data.get("title", "Untitled"),
            narrative_type=narrative_type,
            pattern_summary=n_data.get("pattern_summary", ""),
            confidence=n_data.get("confidence", 0.7),
            prediction=n_data.get("prediction", ""),
            expected_impact=n_data.get("expected_impact", 0),
            trigger_to_watch=n_data.get("trigger_to_watch", ""),
            reversal_risk=n_data.get("reversal_risk", ""),
            recommended_action=n_data.get("recommended_action", ""),
            action_urgency=urgency,
            action_rationale=n_data.get("action_rationale", ""),
        ))
    
    # Build red team insight
    red_team_insight = ""
    if red_team_output:
        vuln = red_team_output.get("primary_vulnerability", "")
        attack = red_team_output.get("attack_vector", "")
        defense = red_team_output.get("defensive_recommendation", "")
        if vuln:
            red_team_insight = f"Vulnerability: {vuln}\nAttack Vector: {attack}\nDefense: {defense}"
    
    # Calculate totals
    total_opp = sum(n.expected_impact for n in narratives if n.expected_impact > 0)
    total_risk = abs(sum(n.expected_impact for n in narratives if n.expected_impact < 0))
    
    # Extract editor info
    killed_count = editor_output.get("killed_count", 0)
    kill_reasons = editor_output.get("kill_reasons", [])
    
    return DailyBrief(
        generated_at=datetime.now(),
        project_id=project_id,
        world_context=world_context,
        journal_context=journal_context,
        market_summary=oracle_output.get("market_summary", ""),
        narratives=narratives,
        red_team_insight=red_team_insight,
        editor_killed_count=killed_count,
        editor_kill_reasons=kill_reasons,
        product_identity=product_identity,
        total_opportunity_value=total_opp,
        total_risk_value=total_risk,
        key_risks=oracle_output.get("key_risks", []),
        key_opportunities=oracle_output.get("key_opportunities", []),
        event_count=len(events),
    )


def _create_fallback_brief(events: List[EnrichedEvent], project_id: str) -> DailyBrief:
    """Create a fallback brief when LLM is unavailable."""
    summary = get_event_summary(events)
    
    return DailyBrief(
        generated_at=datetime.now(),
        project_id=project_id,
        market_summary=f"Analyzed {summary.get('total', 0)} events. LLM unavailable for deep analysis.",
        narratives=[],
        event_count=summary.get("total", 0),
    )


# =============================================================================
# SYNCHRONOUS WRAPPER
# =============================================================================

def run_sherlock_sync(
    events: List[EnrichedEvent],
    project_id: str,
    your_brand: str = "",
    competitor_brand: str = "",
    **kwargs
) -> DailyBrief:
    """Synchronous wrapper for run_sherlock_analysis."""
    return asyncio.run(run_sherlock_analysis(
        events=events,
        project_id=project_id,
        your_brand=your_brand,
        competitor_brand=competitor_brand,
        **kwargs
    ))
