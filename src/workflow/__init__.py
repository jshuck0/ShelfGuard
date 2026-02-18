"""
Workflow ShelfGuard MVP

The workflow layer transforms raw TriggerEvents into actionable Episodes
for the Weekly War-Room Memo.

Pipeline:
    TriggerEvents → Episodes → Top 10 Selection → Memo Rendering

Key modules:
    - episode_builder: Cluster events into episodes
    - reason_codes: Deterministic classification
    - action_templates: Standardized action library
    - memo_renderer: Top 10 selection and formatting
    - alerts: High-signal episode filtering
    - scoreboard: Monthly metrics aggregation
"""

from src.workflow.reason_codes import (
    ReasonCode,
    ReasonCodeConfig,
    get_reason_code_config,
    map_event_type_to_reason_code,
    calculate_severity,
    calculate_confidence,
    REASON_CODE_CONFIGS,
)

from src.workflow.episode_builder import (
    build_episodes,
    cluster_events_by_asin_week,
)

from src.workflow.action_templates import (
    ActionTemplate,
    get_action_template,
    get_all_templates,
    ACTION_TEMPLATES,
)

from src.workflow.memo_renderer import (
    MemoItem,
    WeeklyMemo,
    render_weekly_memo,
    select_top_episodes,
)

from src.workflow.alerts import (
    Alert,
    generate_alerts,
    filter_high_severity,
)

from src.workflow.episode_scoreboard import (
    Scoreboard,
    calculate_scoreboard,
)

__all__ = [
    # Reason Codes
    "ReasonCode",
    "ReasonCodeConfig",
    "get_reason_code_config",
    "map_event_type_to_reason_code",
    "calculate_severity",
    "calculate_confidence",
    "REASON_CODE_CONFIGS",
    # Episode Builder
    "build_episodes",
    "cluster_events_by_asin_week",
    # Action Templates
    "ActionTemplate",
    "get_action_template",
    "get_all_templates",
    "ACTION_TEMPLATES",
    # Memo Renderer
    "MemoItem",
    "WeeklyMemo",
    "render_weekly_memo",
    "select_top_episodes",
    # Alerts
    "Alert",
    "generate_alerts",
    "filter_high_severity",
    # Scoreboard
    "Scoreboard",
    "calculate_scoreboard",
]
