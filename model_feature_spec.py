"""Explicit feature contract for curated Bayesian model inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_INPUTS_DIR = OUTPUT_DIR / "model_inputs"
SEQUENTIAL_MODEL_INPUT_CSV = MODEL_INPUTS_DIR / "sequential_ranking_input.csv"
NON_RUNNER_EVENTS_CSV = OUTPUT_DIR / "non_runner_events.csv"

ID_COLUMNS = [
    "race_id",
    "race_date",
    "race_datetime",
    "runner_index",
    "horse_name",
    "jockey_name",
    "trainer_name",
    "owner_name",
]

CONTEXT_COLUMNS = [
    "racecourse",
    "race_going",
]

TARGET_COLUMNS = [
    "finish_position",
    "won",
    "DNF",
    "WD",
    "NR",
]

LATENT_ENTITY_COLUMNS = [
    "horse_name",
    "jockey_name",
    "trainer_name",
    "owner_name",
]

INTERACTION_COLUMNS = [
    "horse_course_key",
    "horse_going_key",
    "horse_jockey_key",
]

NON_RUNNER_HISTORY_FEATURE_COLUMNS = [
    "horse_prior_non_runner_count",
    "horse_prior_injury_non_runner_count",
    "horse_prior_going_non_runner_count",
    "horse_days_since_last_non_runner",
    "horse_days_since_last_injury_non_runner",
    "horse_prior_dnf_count",
    "horse_days_since_last_dnf",
]

NUMERIC_FEATURE_COLUMNS = [
    "current_mark",
    "draw_number",
    *NON_RUNNER_HISTORY_FEATURE_COLUMNS,
]

FORBIDDEN_COLUMNS = [
    "fixture_timestamp",
    "primary_going",
    "secondary_going",
    "race_name_raw",
    "winner_info_raw",
    "horse_url",
    "jockey_url",
    "trainer_url",
    "horse_jockey_raw",
    "trainer_owner_raw",
    "distance_time_raw",
    "distance_raw",
    "finish_time_raw",
    "position_rank",
    "position_raw",
    "race_distance_raw",
    "race_time_raw",
    "race_runners",
    "race_prize_money",
    "race_age_band",
    "race_type_detail",
    "race_handicapper",
    "race_rating_band",
    "race_min_weight",
    "race_weights_raised",
    "race_rider_type",
    "current_mark_surface",
    "starting_prob",
    "speed",
    "log_speed",
]


def ordered_unique(columns: list[str]) -> list[str]:
    """Preserve column order while removing duplicates."""
    seen: set[str] = set()
    unique_columns: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        unique_columns.append(column)
    return unique_columns


def allowed_model_input_columns() -> list[str]:
    """Return the full ordered allow-list for the curated model input."""
    return ordered_unique(
        [
            *ID_COLUMNS,
            *CONTEXT_COLUMNS,
            *TARGET_COLUMNS,
            *LATENT_ENTITY_COLUMNS,
            *INTERACTION_COLUMNS,
            *NUMERIC_FEATURE_COLUMNS,
        ],
    )


def predictor_columns() -> list[str]:
    """Return the columns the ranking model may use as predictors."""
    return ordered_unique(
        [
            *LATENT_ENTITY_COLUMNS,
            *CONTEXT_COLUMNS,
            *INTERACTION_COLUMNS,
            *NUMERIC_FEATURE_COLUMNS,
        ],
    )


def validate_model_input_frame(df: pd.DataFrame) -> None:
    """Ensure curated model input contains only allowed columns."""
    allowed = set(allowed_model_input_columns())
    required = set(ID_COLUMNS + CONTEXT_COLUMNS + TARGET_COLUMNS + INTERACTION_COLUMNS)
    unexpected = sorted(set(df.columns) - allowed)
    missing = sorted(required - set(df.columns))
    forbidden_present = sorted(set(df.columns).intersection(FORBIDDEN_COLUMNS))

    if unexpected:
        msg = f"Curated model input includes unexpected columns: {unexpected}"
        raise ValueError(msg)
    if missing:
        msg = f"Curated model input is missing required columns: {missing}"
        raise ValueError(msg)
    if forbidden_present:
        msg = f"Curated model input includes forbidden columns: {forbidden_present}"
        raise ValueError(msg)
