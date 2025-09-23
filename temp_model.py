# %%
"""Lightweight hierarchical Bayesian model for race speed (PyMC v4)."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "model"

NUM_COLS = [
    "going_stick",
    "soil_moisture_pct",
    "draw_number",
    "handicap_ran_off",
    "current_mark",
]


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean and split races into training and test sets."""
    race_raw = pd.read_csv(OUTPUT_DIR / "race_df.csv")

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
        .assign(
            speed=lambda df: df["race_distance_m"] / df["finish_time_sec"],
            log_speed=lambda df: np.log(df["race_distance_m"] / df["finish_time_sec"]),
        )
        .dropna(
            subset=[
                "speed",
                "horse_name",
                "jockey_name",
                "finish_time_sec",
                "current_mark_surface",
            ],
        )
    ).assign(
        won=lambda df: df.groupby("race_id")["speed"].transform(lambda x: x == x.max()),
    )

    # Median imputation
    for col in ["going_stick", "soil_moisture_pct", "draw_number", "current_mark"]:
        med = flat[col].median()
        flat[col] = flat[col].fillna(med)

    # Train/test split by fixture
    race_ids = sorted(flat["race_id"].unique())
    cutoff = int(len(race_ids) * 0.75)
    train_ids, test_ids = race_ids[:cutoff], race_ids[cutoff:]
    train_df = flat[flat["race_id"].isin(train_ids)].copy()
    test_df = flat[flat["race_id"].isin(test_ids)].copy()
    return train_df, test_df


# ---------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------


def effect(name: str, n: int, sigma_scale: float = 0.5) -> pt.TensorVariable:
    """Hierarchical effect with sum-to-zero non-centred parameterisation."""
    raw = pm.Normal(f"{name}_raw", 0.0, 1.0, shape=n)
    sigma = pm.HalfNormal(f"sigma_{name}", sigma_scale)
    return (raw - pt.mean(raw)) * sigma


def safe_codes(series: pd.Series, cats: pd.Index) -> np.ndarray:
    """Encode categories with unseen levels mapped to 0 (average effect)."""
    cat = pd.Categorical(series, categories=cats)
    return np.where(cat.codes == -1, 0, cat.codes)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------


def fit_model(
    x_train: np.ndarray,
    y_train_z: np.ndarray,
    horses: pd.Categorical,
    jockeys: pd.Categorical,
    courses: pd.Categorical,
    weathers: pd.Categorical,
    surfs: pd.Categorical,
    classes: pd.Categorical,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and fit the Bayesian model."""
    horse_idx = horses.codes
    jock_idx = jockeys.codes
    course_idx = courses.codes
    weather_idx = weathers.codes
    surf_idx = surfs.codes
    class_idx = classes.codes

    with pm.Model() as model:
        # Data containers
        x = pm.Data("x", x_train)
        horse_d = pm.Data("horse_idx", horse_idx)
        jock_d = pm.Data("jock_idx", jock_idx)
        course_d = pm.Data("course_idx", course_idx)
        weather_d = pm.Data("weather_idx", weather_idx)
        surf_d = pm.Data("surf_idx", surf_idx)
        class_d = pm.Data("class_idx", class_idx)

        # Priors
        intercept = pm.Normal("intercept", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0, shape=x.shape[1])

        # Linear predictor with group effects
        mu = (
            intercept
            + pt.dot(x, beta)
            + effect("horse", horses.categories.size, sigma_scale=1.0)[horse_d]
            + effect("jock", jockeys.categories.size, sigma_scale=1.0)[jock_d]
            + effect("course", courses.categories.size)[course_d]
            + effect("weather", weathers.categories.size)[weather_d]
            + effect("surf", surfs.categories.size)[surf_d]
            + effect("class", classes.categories.size)[class_d]
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

    out_dir = OUTPUT_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

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
    summary.to_csv(out_dir / "race_summary.csv", index=False)
    logger.info("Saved race summary to {}", out_dir / "race_summary.csv")

    roi = (summary["pred_winner_return"].sum() - len(summary)) / len(summary)
    logger.info("ROI from 1-unit bets: {:.2%}", roi)


def main() -> None:
    """Train the model, generate predictions and evaluate performance."""
    train_df, test_df = prepare_data()

    # Standardise continuous predictors
    scaler_x = StandardScaler().fit(train_df[NUM_COLS])
    x_train = scaler_x.transform(train_df[NUM_COLS])
    x_test = scaler_x.transform(test_df[NUM_COLS])

    # Standardise target
    y_train = train_df["log_speed"].to_numpy()
    y_mean, y_std = y_train.mean(), y_train.std(ddof=0)
    y_train_z = (y_train - y_mean) / y_std

    # Encode categories (train set)
    horses = pd.Categorical(train_df["horse_name"])
    jockeys = pd.Categorical(train_df["jockey_name"])
    courses = pd.Categorical(train_df["racecourse"])
    weathers = pd.Categorical(train_df["weather_category"])
    surfs = pd.Categorical(train_df["current_mark_surface"])
    classes = pd.Categorical(train_df["class"].astype(str))

    # Fit model
    model, idata = fit_model(x_train, y_train_z, horses, jockeys, courses, weathers, surfs, classes)

    # Predict on test set
    with model:
        pm.set_data(
            {
                "x": x_test,
                "horse_idx": safe_codes(test_df["horse_name"], horses.categories),
                "jock_idx": safe_codes(test_df["jockey_name"], jockeys.categories),
                "course_idx": safe_codes(test_df["racecourse"], courses.categories),
                "weather_idx": safe_codes(test_df["weather_category"], weathers.categories),
                "surf_idx": safe_codes(test_df["current_mark_surface"], surfs.categories),
                "class_idx": safe_codes(test_df["class"].astype(str), classes.categories),
            },
        )
        mu_pred = pm.sample_posterior_predictive(idata, var_names=["mu"], predictions=True)

    mu_mean = mu_pred.predictions["mu"].mean(("chain", "draw")).to_numpy()
    preds = np.exp(y_mean + y_std * mu_mean)

    # Evaluate
    evaluate_predictions(test_df, preds)


# %%
if __name__ == "__main__":
    main()
# %%
