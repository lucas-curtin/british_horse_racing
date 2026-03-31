"""Curated data loader for sequential Bayesian ranking models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from model_feature_spec import (
    FORBIDDEN_COLUMNS,
    SEQUENTIAL_MODEL_INPUT_CSV,
    allowed_model_input_columns,
    predictor_columns,
    validate_model_input_frame,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_model_input(path: Path = SEQUENTIAL_MODEL_INPUT_CSV) -> pd.DataFrame:
    """Load the curated sequential model input and enforce the feature contract."""
    model_input = pd.read_csv(path, parse_dates=["race_date", "race_datetime"])
    validate_model_input_frame(model_input)
    return model_input.sort_values(
        ["race_datetime", "race_id", "runner_index"],
        kind="stable",
    ).reset_index(drop=True)


def model_predictor_frame(model_input: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns allowed to enter the ranking model."""
    missing_predictors = sorted(set(predictor_columns()) - set(model_input.columns))
    if missing_predictors:
        msg = f"Curated model input is missing predictor columns: {missing_predictors}"
        raise ValueError(msg)
    forbidden_present = sorted(set(predictor_columns()).intersection(FORBIDDEN_COLUMNS))
    if forbidden_present:
        msg = f"Predictor list includes forbidden columns: {forbidden_present}"
        raise ValueError(msg)
    return model_input.loc[:, predictor_columns()].copy()


def main() -> None:
    """Load and summarise the curated sequential model input."""
    model_input = load_model_input()
    logger.info("Loaded {} model-input rows from {}", len(model_input), SEQUENTIAL_MODEL_INPUT_CSV)
    logger.info("Allowed columns: {}", allowed_model_input_columns())
    logger.info("Predictor columns: {}", predictor_columns())


if __name__ == "__main__":
    main()
