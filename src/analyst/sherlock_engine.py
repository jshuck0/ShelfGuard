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

ANALYST_PROMPT = """You are a market analyst examining 90 days of market events for an Amazon seller.

{world_context}

{journal_context}

{event_stream}

=== YOUR TASK ===
Find the 5 most important PATTERNS in this event stream.

For each pattern:
1. Describe WHAT happened (specific events, dates, products)
2. Identify CHAIN REACTIONS (did Event A consistently precede Event B?)
3. Quantify the IMPACT (revenue, rank, share)
4. Note which products are YOURS (📍YOUR) vs COMPETITORS (🎯COMP)

Focus on:
- Competitor OOS events = conquest opportunities
- Price movements that preceded rank changes
- Review surges that drove sales
- Buy Box shifts between sellers

Output as JSON:
{{
    "patterns": [
        {{
            "title": "Brief title",
            "description": "What happened",
            "chain_reaction": "Event A led to Event B",
            "products_involved": ["ASIN1", "ASIN2"],
            "your_products_affected": true,
            "estimated_impact": "$X or X%",
            "dates": "Jan 15-20"
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

ORACLE_PROMPT = """You are a strategic advisor for an Amazon seller. Based on verified patterns, predict what will happen and what to do.

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


async def run_oracle_step(
    client: AsyncOpenAI,
    verified_patterns: Dict,
    world_context: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Step 3: Oracle predicts outcomes and recommends actions."""
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
    
    # Run the 4-step loop
    print("🔍 Step 1: Analyst finding patterns...")
    analyst_output = await run_analyst_step(
        client, events, world_context_str, journal_context_str, model
    )
    
    print("🤔 Step 2: Skeptic verifying causality...")
    skeptic_output = await run_skeptic_step(client, analyst_output, model)
    
    print("🔮 Step 3: Oracle predicting outcomes...")
    oracle_output = await run_oracle_step(
        client, skeptic_output, world_context_str, model
    )
    
    print("⚔️ Step 4: Red Team finding vulnerabilities...")
    red_team_output = await run_red_team_step(
        client, events, your_brand, competitor_brand, model
    )
    
    # Build the Daily Brief
    brief = _build_daily_brief(
        oracle_output=oracle_output,
        red_team_output=red_team_output,
        world_context=world_context_dict,
        journal_context=journal_context_str,
        events=events,
        project_id=project_id,
    )
    
    return brief


def _build_daily_brief(
    oracle_output: Dict,
    red_team_output: Dict,
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
    
    return DailyBrief(
        generated_at=datetime.now(),
        project_id=project_id,
        world_context=world_context,
        journal_context=journal_context,
        market_summary=oracle_output.get("market_summary", ""),
        narratives=narratives,
        red_team_insight=red_team_insight,
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
