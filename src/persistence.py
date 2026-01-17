"""
ShelfGuard Persistence Layer
==============================
Phase 2: Pin to State & Mission Profiles

This module handles:
1. Creating new projects from discovered ASINs
2. Managing Mission Profiles (Bodyguard, Scout, Surgeon)
3. Row Level Security (RLS) scoping to user
"""

import pandas as pd
import streamlit as st
from supabase import Client
from typing import Dict, List, Optional
from datetime import datetime
import uuid


def create_supabase_client() -> Client:
    """
    Initialize Supabase client using Streamlit secrets.

    Returns: Authenticated Supabase client
    """
    from supabase import create_client
    return create_client(st.secrets["url"], st.secrets["key"])


def pin_to_state(
    asins: List[str],
    project_name: str,
    mission_type: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Create a new project (state) from pruned ASINs.

    Args:
        asins: List of ASINs from the discovery phase
        project_name: User-provided name for the project
        mission_type: One of ["bodyguard", "scout", "surgeon"]
        user_id: Auth user ID (for RLS). If None, uses anonymous mode
        metadata: Additional context (search query, stats, etc.)

    Returns:
        project_id (UUID string)

    Database Schema Required:
        Table: projects
        Columns:
            - id (uuid, primary key)
            - created_at (timestamp)
            - user_id (uuid, foreign key to auth.users, RLS)
            - project_name (text)
            - mission_type (text)
            - asin_count (int)
            - metadata (jsonb)
    """
    supabase = create_supabase_client()

    # Generate project ID
    project_id = str(uuid.uuid4())

    # Build project record
    project_data = {
        "id": project_id,
        "created_at": datetime.utcnow().isoformat(),
        "user_id": user_id,  # Will be null for anonymous users
        "project_name": project_name,
        "mission_type": mission_type,
        "asin_count": len(asins),
        "metadata": metadata or {}
    }

    try:
        # Insert project
        result = supabase.table("projects").insert(project_data).execute()

        # Insert tracked ASINs
        tracked_asins = [
            {
                "project_id": project_id,
                "asin": asin,
                "added_at": datetime.utcnow().isoformat(),
                "is_active": True
            }
            for asin in asins
        ]

        supabase.table("tracked_asins").insert(tracked_asins).execute()

        return project_id

    except Exception as e:
        st.error(f"❌ Failed to create project: {str(e)}")
        raise


def get_mission_profile_config(mission_type: str) -> Dict:
    """
    Returns alert priority configuration for each Mission Profile.

    Mission Types:
    1. "bodyguard" - Defensive focus (price protection, Buy Box monitoring)
    2. "scout" - Offensive focus (new entrants, rising stars)
    3. "surgeon" - Efficiency focus (review gaps, ad waste)

    Returns: Dict with priority weights and alert thresholds
    """
    profiles = {
        "bodyguard": {
            "name": "🛡️ The Bodyguard",
            "focus": "Defensive",
            "priorities": {
                "price_undercut": 1.0,     # Highest priority
                "buybox_loss": 1.0,
                "competitor_surge": 0.8,
                "new_entrants": 0.5,
                "review_gaps": 0.3,
                "ad_waste": 0.3
            },
            "thresholds": {
                "price_delta_alert": -0.10,  # Alert if competitor is 10% cheaper
                "buybox_share_alert": 0.50,  # Alert if BB share drops below 50%
            }
        },
        "scout": {
            "name": "🔍 The Scout",
            "focus": "Offensive",
            "priorities": {
                "new_entrants": 1.0,
                "rising_stars": 1.0,        # ASINs with BSR improvement >20%
                "competitor_surge": 0.8,
                "price_undercut": 0.5,
                "buybox_loss": 0.4,
                "review_gaps": 0.3,
                "ad_waste": 0.2
            },
            "thresholds": {
                "bsr_improvement_alert": 0.20,  # Alert if BSR improved 20%+ in 7 days
                "new_entrant_bsr_threshold": 50000,  # Track new ASINs with BSR < 50k
            }
        },
        "surgeon": {
            "name": "🔬 The Surgeon",
            "focus": "Efficiency",
            "priorities": {
                "review_gaps": 1.0,
                "ad_waste": 1.0,
                "pricing_inefficiency": 0.8,
                "buybox_loss": 0.5,
                "price_undercut": 0.4,
                "new_entrants": 0.2,
                "competitor_surge": 0.2
            },
            "thresholds": {
                "review_gap_pct": 0.50,  # Alert if reviews < 50% of category avg
                "ad_efficiency_min": 2.0,  # Alert if ROAS < 2.0
            }
        }
    }

    return profiles.get(mission_type, profiles["bodyguard"])


def load_user_projects(user_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load all projects for a given user.

    Args:
        user_id: Auth user ID (None for anonymous mode)

    Returns:
        DataFrame with projects, sorted by created_at desc
    """
    supabase = create_supabase_client()

    try:
        # Query projects (RLS will automatically filter by user_id if configured)
        query = supabase.table("projects").select("*")

        if user_id:
            query = query.eq("user_id", user_id)

        result = query.order("created_at", desc=True).execute()

        if not result.data:
            return pd.DataFrame()

        return pd.DataFrame(result.data)

    except Exception as e:
        st.error(f"❌ Failed to load projects: {str(e)}")
        return pd.DataFrame()


def load_project_asins(project_id: str) -> List[str]:
    """
    Load all tracked ASINs for a specific project.

    Args:
        project_id: UUID of the project

    Returns:
        List of ASINs
    """
    supabase = create_supabase_client()

    try:
        result = supabase.table("tracked_asins").select("asin").eq(
            "project_id", project_id
        ).eq("is_active", True).execute()

        if not result.data:
            return []

        return [row["asin"] for row in result.data]

    except Exception as e:
        st.error(f"❌ Failed to load project ASINs: {str(e)}")
        return []


def update_project_metadata(project_id: str, metadata_updates: Dict) -> bool:
    """
    Update project metadata (e.g., last_analysis_date, total_revenue).

    Args:
        project_id: UUID of the project
        metadata_updates: Dict of key-value pairs to merge into metadata

    Returns:
        Success boolean
    """
    supabase = create_supabase_client()

    try:
        # Fetch current metadata
        result = supabase.table("projects").select("metadata").eq(
            "id", project_id
        ).execute()

        if not result.data:
            return False

        current_metadata = result.data[0].get("metadata", {})

        # Merge updates
        updated_metadata = {**current_metadata, **metadata_updates}

        # Update
        supabase.table("projects").update(
            {"metadata": updated_metadata}
        ).eq("id", project_id).execute()

        return True

    except Exception as e:
        st.error(f"❌ Failed to update project metadata: {str(e)}")
        return False
