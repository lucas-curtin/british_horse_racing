"""Canonical model entrypoint for the repo.

For now this script keeps the current best available batch trainer/evaluator
while also validating the curated sequential-model input contract at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from model_feature_spec import (
    SEQUENTIAL_MODEL_INPUT_CSV,
    allowed_model_input_columns,
    predictor_columns,
    validate_model_input_frame,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "model"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FEATURES_CSV = OUTPUT_DIR / "historical_features.csv"

BASE_NUM_COLS = [
    "going_stick",
    "soil_moisture_pct",
    "draw_number",
    "current_mark",
    "starting_prob",
    "horse_prior_runs",
    "horse_prior_wins",
    "horse_prior_places",
    "horse_prior_win_rate",
    "horse_prior_place_rate",
    "horse_prior_avg_finish_position",
    "horse_prior_avg_speed",
    "horse_prior_avg_finish_time_sec",
    "horse_prior_course_runs",
    "horse_prior_course_wins",
    "horse_prior_going_runs",
    "horse_prior_going_wins",
    "horse_prior_distance_bucket_runs",
    "horse_prior_distance_bucket_wins",
    "horse_prior_mark_change",
    "horse_prior_days_since_last_run",
    "jockey_prior_rides",
    "jockey_prior_wins",
    "jockey_prior_places",
    "jockey_prior_win_rate",
    "jockey_prior_place_rate",
    "jockey_prior_avg_finish_position",
    "jockey_prior_avg_speed",
    "jockey_prior_avg_finish_time_sec",
    "jockey_prior_course_runs",
    "jockey_prior_course_wins",
    "jockey_prior_going_runs",
    "jockey_prior_going_wins",
    "jockey_prior_distance_bucket_runs",
    "jockey_prior_distance_bucket_wins",
    "jockey_prior_days_since_last_ride",
    "jockey_prior_days_since_last_win",
    "trainer_prior_runs",
    "trainer_prior_wins",
    "trainer_prior_places",
    "trainer_prior_win_rate",
    "owner_prior_runs",
    "owner_prior_wins",
    "owner_prior_places",
    "owner_prior_win_rate",
    "pair_prior_rides",
    "pair_prior_wins",
    "pair_prior_places",
    "pair_prior_win_rate",
]

EMPTY_SPLIT_ERROR = (
    "Train/test split is empty. Widen the scrape date range so the model has enough races."
)
NO_NUMERIC_PREDICTORS_ERROR = (
    "No usable numeric predictors available after filtering missing columns."
)
NAN_PREDICTORS_ERROR = "Numeric predictors still contain NaNs after imputation."
ZERO_VARIANCE_ERROR = (
    "Training target has zero or undefined variance; widen the scrape date range."
)


@dataclass(frozen=True)
class CategoryData:
    """Categorical training encodings used by the hierarchical model."""

    horses: pd.Categorical
    jockeys: pd.Categorical
    courses: pd.Categorical
    weathers: pd.Categorical
    surfs: pd.Categorical
    classes: pd.Categorical


def load_curated_model_input(
    path: Path = SEQUENTIAL_MODEL_INPUT_CSV,
) -> pd.DataFrame:
    """Load and validate the curated sequential-model input."""
    if not path.exists():
        msg = f"Missing curated model input at {path}. Run `python preprocessing.py` first."
        raise FileNotFoundError(msg)

    model_input = pd.read_csv(path, parse_dates=["race_date", "race_datetime"])
    validate_model_input_frame(model_input)
    return model_input.sort_values(
        ["race_datetime", "race_id", "runner_index"],
        kind="stable",
    ).reset_index(drop=True)


def log_curated_model_input_summary(model_input: pd.DataFrame) -> None:
    """Log the curated input contract that future sequential training will use."""
    logger.info(
        "Validated curated sequential input: {} rows, {} allowed columns.",
        len(model_input),
        len(allowed_model_input_columns()),
    )
    logger.info("Sequential predictor contract: {}", predictor_columns())


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load, clean and split races into training and test sets."""
    if not FEATURES_CSV.exists():
        msg = f"Missing historical features at {FEATURES_CSV}. Run `python preprocessing.py` first."
        raise FileNotFoundError(msg)

    race_raw = pd.read_csv(FEATURES_CSV, parse_dates=["race_date", "race_datetime"])

    race_types = [
        "handicap",
        "steeple",
        "chase",
        "novice",
        "hurdle",
        "maiden",
        "national_hunt",
        "selling",
    ]

    flat = (
        race_raw.loc[
            (race_raw["handicap"])
            & (~race_raw[[c for c in race_types if c != "handicap"]].any(axis=1))
        ]
        .loc[(~race_raw["WD"]) & (~race_raw["NR"]) & (~race_raw["DNF"])]
        .dropna(
            subset=[
                "log_speed",
                "horse_name",
                "jockey_name",
                "current_mark_surface",
            ],
        )
    )

    for col in [
        "going_stick",
        "soil_moisture_pct",
        "draw_number",
        "current_mark",
        "starting_prob",
    ]:
        med = flat[col].median()
        flat[col] = flat[col].fillna(med)

    ordered_races = (
        flat[["race_id", "race_datetime"]]
        .drop_duplicates()
        .sort_values(["race_datetime", "race_id"], kind="stable")
    )
    race_ids = ordered_races["race_id"].tolist()
    cutoff = int(len(race_ids) * 0.75)
    train_ids, test_ids = race_ids[:cutoff], race_ids[cutoff:]
    train_df = flat[flat["race_id"].isin(train_ids)].copy()
    test_df = flat[flat["race_id"].isin(test_ids)].copy()

    if train_df.empty or test_df.empty:
        msg = EMPTY_SPLIT_ERROR
        raise ValueError(msg)

    usable_num_cols: list[str] = []
    for col in BASE_NUM_COLS:
        train_non_null = train_df[col].notna().sum()
        if train_non_null == 0:
            logger.warning("Dropping numeric feature '{}' because it has no training data.", col)
            continue
        usable_num_cols.append(col)

    if not usable_num_cols:
        msg = NO_NUMERIC_PREDICTORS_ERROR
        raise ValueError(msg)

    medians = train_df[usable_num_cols].median().fillna(0.0)
    train_df[usable_num_cols] = train_df[usable_num_cols].fillna(medians)
    test_df[usable_num_cols] = test_df[usable_num_cols].fillna(medians)

    if train_df[usable_num_cols].isna().any().any() or test_df[usable_num_cols].isna().any().any():
        msg = NAN_PREDICTORS_ERROR
        raise ValueError(msg)

    return train_df, test_df, usable_num_cols


def effect(name: str, n: int, sigma_scale: float = 0.5) -> pt.TensorVariable:
    """Hierarchical effect with sum-to-zero non-centred parameterisation."""
    raw = pm.Normal(f"{name}_raw", 0.0, 1.0, shape=n)
    sigma = pm.HalfNormal(f"sigma_{name}", sigma_scale)
    return (raw - pt.mean(raw)) * sigma


def safe_codes(series: pd.Series, cats: pd.Index) -> np.ndarray:
    """Encode categories with unseen levels mapped to 0 (average effect)."""
    cat = pd.Categorical(series, categories=cats)
    return np.where(cat.codes == -1, 0, cat.codes)


def fit_model(
    x_train: np.ndarray,
    y_train_z: np.ndarray,
    categories: CategoryData,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and fit the current working hierarchical model."""
    horse_idx = categories.horses.codes
    jock_idx = categories.jockeys.codes
    course_idx = categories.courses.codes
    weather_idx = categories.weathers.codes
    surf_idx = categories.surfs.codes
    class_idx = categories.classes.codes

    with pm.Model() as model:
        x = pm.Data("x", x_train)
        horse_d = pm.Data("horse_idx", horse_idx)
        jock_d = pm.Data("jock_idx", jock_idx)
        course_d = pm.Data("course_idx", course_idx)
        weather_d = pm.Data("weather_idx", weather_idx)
        surf_d = pm.Data("surf_idx", surf_idx)
        class_d = pm.Data("class_idx", class_idx)

        intercept = pm.Normal("intercept", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0, shape=x.shape[1])

        mu = (
            intercept
            + pt.dot(x, beta)
            + effect("horse", categories.horses.categories.size, sigma_scale=1.0)[horse_d]
            + effect("jock", categories.jockeys.categories.size, sigma_scale=1.0)[jock_d]
            + effect("course", categories.courses.categories.size)[course_d]
            + effect("weather", categories.weathers.categories.size)[weather_d]
            + effect("surf", categories.surfs.categories.size)[surf_d]
            + effect("class", categories.classes.categories.size)[class_d]
        )

        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_train_z)
        pm.Deterministic("mu", mu)

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.95,
            max_treedepth=12,
            random_seed=451,
            return_inferencedata=True,
        )

    logger.info(az.summary(idata, var_names=["intercept", "sigma"]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, MODEL_DIR / "model_fit.nc")
    return model, idata


def evaluate_predictions(test_df: pd.DataFrame, preds: np.ndarray) -> None:
    """Evaluate predictions: accuracy, top-3 accuracy and ROI."""
    pred_df = (
        test_df.assign(pred_speed=preds)
        .assign(
            pred_rank=lambda d: d.groupby("race_id")["pred_speed"]
            .rank(ascending=False, method="first")
            .astype(int),
        )
        .assign(pred_winner=lambda d: d["pred_rank"].eq(1))
        .assign(actual_winner=lambda d: d["won"].astype(bool))
        .assign(decimal_odds=lambda d: 1.0 / d["starting_prob"].replace(0, np.nan))
        .assign(bet_return=lambda d: np.where(d["actual_winner"], d["decimal_odds"], 0.0))
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False)

    winners = pred_df.loc[pred_df.groupby("race_id")["pred_speed"].idxmax()]
    actual_winners = pred_df.loc[pred_df.groupby("race_id")["speed"].idxmax()]
    acc = accuracy_score(actual_winners["horse_name"], winners["horse_name"])
    logger.info("Winner accuracy: {:.2%}", acc)

    top3_acc = np.mean(
        [
            g.loc[g["speed"].idxmax(), "horse_name"]
            in g.sort_values("pred_speed", ascending=False)["horse_name"].head(3).to_numpy()
            for _, g in pred_df.groupby("race_id")
        ],
    )
    logger.info("Top-3 accuracy: {:.2%}", top3_acc)

    summary = (
        pred_df.groupby("race_id")
        .apply(
            lambda g: pd.Series(
                {
                    "racecourse": g["racecourse"].iloc[0],
                    "date": g["race_date"].iloc[0],
                    "n_runners": len(g),
                    "pred_winner": g.loc[g["pred_rank"].eq(1), "horse_name"].iloc[0],
                    "actual_winner": g.loc[g["actual_winner"], "horse_name"].iloc[0],
                    "pred_winner_return": (
                        g.loc[g["pred_rank"].eq(1), "decimal_odds"].iloc[0]
                        if g.loc[g["pred_rank"].eq(1), "actual_winner"].iloc[0]
                        else 0.0
                    ),
                },
            ),
        )
        .reset_index()
    )
    summary.to_csv(PREDICTIONS_DIR / "race_summary.csv", index=False)
    logger.info("Saved race summary to {}", PREDICTIONS_DIR / "race_summary.csv")

    roi = (summary["pred_winner_return"].sum() - len(summary)) / len(summary)
    logger.info("ROI from 1-unit bets: {:.2%}", roi)


def main() -> None:
    """Run the canonical model workflow."""
    curated_model_input = load_curated_model_input()
    log_curated_model_input_summary(curated_model_input)

    train_df, test_df, num_cols = prepare_data()
    logger.info("Using numeric predictors: {}", num_cols)

    scaler_x = StandardScaler().fit(train_df[num_cols])
    x_train = scaler_x.transform(train_df[num_cols])
    x_test = scaler_x.transform(test_df[num_cols])

    y_train = train_df["log_speed"].to_numpy()
    y_mean, y_std = y_train.mean(), y_train.std(ddof=0)
    if np.isnan(y_std) or y_std == 0:
        msg = ZERO_VARIANCE_ERROR
        raise ValueError(msg)
    y_train_z = (y_train - y_mean) / y_std

    categories = CategoryData(
        horses=pd.Categorical(train_df["horse_name"]),
        jockeys=pd.Categorical(train_df["jockey_name"]),
        courses=pd.Categorical(train_df["racecourse"]),
        weathers=pd.Categorical(train_df["weather_category"]),
        surfs=pd.Categorical(train_df["current_mark_surface"]),
        classes=pd.Categorical(train_df["class"].astype(str)),
    )

    model, idata = fit_model(x_train, y_train_z, categories)

    with model:
        pm.set_data(
            {
                "x": x_test,
                "horse_idx": safe_codes(test_df["horse_name"], categories.horses.categories),
                "jock_idx": safe_codes(test_df["jockey_name"], categories.jockeys.categories),
                "course_idx": safe_codes(test_df["racecourse"], categories.courses.categories),
                "weather_idx": safe_codes(
                    test_df["weather_category"],
                    categories.weathers.categories,
                ),
                "surf_idx": safe_codes(
                    test_df["current_mark_surface"],
                    categories.surfs.categories,
                ),
                "class_idx": safe_codes(
                    test_df["class"].astype(str),
                    categories.classes.categories,
                ),
            },
        )
        mu_pred = pm.sample_posterior_predictive(idata, var_names=["mu"], predictions=True)

    mu_mean = mu_pred.predictions["mu"].mean(("chain", "draw")).to_numpy()
    preds = np.exp(y_mean + y_std * mu_mean)
    evaluate_predictions(test_df, preds)


if __name__ == "__main__":
    main()
