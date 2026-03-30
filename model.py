# %%
"""Hierarchical Bayesian model for race speed (PyMC v4) with train/test split."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from loguru import logger
from sklearn.metrics import accuracy_score

# Directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "model"

# Numeric predictors
NUM_COLS = [
    "going_stick",
    "soil_moisture_pct",
    "draw_number",
    "handicap_ran_off",
    "current_mark",
]


# ---------------------------------------------------------------------
# Helpers for categorical encoding
# ---------------------------------------------------------------------
def encode_categories(
    df: pd.DataFrame,
    cols: list[str],
    existing_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Encode categorical columns into integer indices.

    Uses an explicit '__UNK__' bucket (neutral effect) for unseen categories.
    If `existing_maps` is provided (for test set), it reuses the train mappings.
    Otherwise, it creates new mappings.
    """
    indices = {}
    maps = {} if existing_maps is None else existing_maps

    for col in cols:
        series = df[col].astype("string")

        if existing_maps is None:
            cats = pd.Index(series.dropna().unique())
            # map known categories 0..n-1, reserve last index for unknowns
            maps[col] = {cat: i for i, cat in enumerate(cats)}
            maps[col]["__UNK__"] = len(cats)
        elif "__UNK__" not in maps[col]:
            # ensure the unknown bucket exists in reused maps
            maps[col]["__UNK__"] = len(maps[col])

        # map values; unseen -> __UNK__
        idx = series.map(maps[col]).fillna(maps[col]["__UNK__"]).astype(int).to_numpy()
        indices[f"{col}_idx"] = idx

    return indices, maps


def sum_to_zero_noncentered(name: str, n: int, sigma_prior: float = 0.5) -> pm.Deterministic:
    """Hierarchical group effect with sum-to-zero non-centred parametrisation."""
    raw = pm.Normal(f"{name}_raw", 0.0, 1.0, shape=n)
    tau = pm.HalfNormal(f"sigma_{name}", sigma_prior)
    eff = (raw - pt.mean(raw)) * tau
    return pm.Deterministic(f"{name}_eff", eff)


def pad_unknown(eff: pt.TensorVariable) -> pt.TensorVariable:
    """Append a neutral (zero) effect as the last element for the '__UNK__' bucket."""
    zero = pt.as_tensor_variable([0.0])
    return pt.concatenate([eff, zero])


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean and split dataset into train and test sets by race_id."""
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
            speed=lambda d: d["race_distance_m"] / d["finish_time_sec"],
            log_speed=lambda d: np.log(d["race_distance_m"] / d["finish_time_sec"]),
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
    ).assign(won=lambda d: d.groupby("race_id")["speed"].transform(lambda x: x == x.max()))

    # Split by race_id
    race_ids = sorted(flat["race_id"].unique())
    cutoff = int(len(race_ids) * 0.75)
    train_ids, test_ids = race_ids[:cutoff], race_ids[cutoff:]

    train_df = flat[flat["race_id"].isin(train_ids)].copy()
    test_df = flat[flat["race_id"].isin(test_ids)].copy()

    # Median imputation for numeric columns (fit on train, apply to both)
    medians = train_df[NUM_COLS].median()
    train_df[NUM_COLS] = train_df[NUM_COLS].fillna(medians)
    test_df[NUM_COLS] = test_df[NUM_COLS].fillna(medians)

    return train_df, test_df


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main() -> None:
    """Train hierarchical model, predict, and evaluate."""
    train_df, test_df = prepare_data()
    logger.info("Training hierarchical Bayesian model...")

    # Encode categorical indices
    cat_cols = [
        "horse_name",
        "jockey_name",
        "racecourse",
        "weather_category",
        "current_mark_surface",
        "class",
    ]
    train_idx, maps = encode_categories(train_df, cat_cols)
    test_idx, _ = encode_categories(test_df, cat_cols, existing_maps=maps)

    # Continuous predictors
    x_train = train_df[NUM_COLS].to_numpy()
    y_train = train_df["log_speed"].to_numpy()

    with pm.Model() as model:
        # Data containers
        y = pm.Data("y", y_train)
        x = pm.Data("x", x_train)
        idx_data = {k: pm.Data(k, v) for k, v in train_idx.items()}

        # Priors
        intercept = pm.Normal("intercept", 0.0, 0.5)
        beta = pm.Normal("beta", 0.0, 0.5, shape=x.shape[1])

        # Group effects
        effs = {}
        for col in cat_cols:
            # exclude the reserved __UNK__ bucket when creating effects
            n_levels_known = len(maps[col]) - 1
            eff = sum_to_zero_noncentered(
                col,
                n_levels_known,
                sigma_prior=1.0 if col in ["horse_name", "jockey_name"] else 0.5,
            )
            # append neutral zero for unknown bucket
            effs[col] = pad_unknown(eff)

        # Linear predictor
        mu = intercept + pt.dot(x, beta)
        for col in cat_cols:
            mu += effs[col][idx_data[f"{col}_idx"]]

        # Likelihood
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        pm.Deterministic("mu", mu)

        # Fit model
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.9,
            random_seed=451,
            return_inferencedata=True,
        )

    logger.info(az.summary(idata, var_names=["intercept", "sigma"]))
    az.to_netcdf(idata, MODEL_DIR / "model_fit.nc")

    # Predictions
    x_test = test_df[NUM_COLS].to_numpy()
    with model:
        pm.set_data({"x": x_test, **dict(test_idx.items())})
        mu_pred = pm.sample_posterior_predictive(idata, var_names=["mu"], predictions=True)

    preds = np.exp(mu_pred.predictions["mu"].mean(("chain", "draw")).values)

    # Evaluation
    pred_df = test_df.assign(pred_speed=preds).assign(
        pred_rank=lambda d: d.groupby("race_id")["pred_speed"]
        .rank(ascending=False, method="first")
        .astype(int),
        pred_winner=lambda d: d["pred_rank"].eq(1),
        actual_winner=lambda d: d["won"].astype(bool),
        decimal_odds=lambda d: 1.0 / d["starting_prob"].replace(0, np.nan),
        bet_return=lambda d: np.where(d["actual_winner"], d["decimal_odds"], 0.0),
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
            grp.loc[grp["speed"].idxmax(), "horse_name"]
            in grp.sort_values("pred_speed", ascending=False)["horse_name"].head(3).to_numpy()
            for _, grp in pred_df.groupby("race_id")
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
                    "pred_winner_return": g.loc[g["pred_rank"].eq(1), "decimal_odds"].iloc[0]
                    if g.loc[g["pred_rank"].eq(1), "actual_winner"].iloc[0]
                    else 0.0,
                },
            ),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "race_summary.csv", index=False)

    roi = (summary["pred_winner_return"].sum() - len(summary)) / len(summary)
    logger.info("ROI from 1-unit bets: {:.2%}", roi)


# %%
if __name__ == "__main__":
    main()
# %%
