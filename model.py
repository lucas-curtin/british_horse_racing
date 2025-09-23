# %%
"""Hierarchical Bayesian model for race speed (PyMC v4) with train/test split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import arviz as az

# Save
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from loguru import logger
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

model_dir = OUTPUT_DIR / "model"


def make_train_map(series: pd.Series) -> dict[str, int]:
    """Return mapping from category to index. NaN is preserved."""
    cats = pd.Index(series.astype("string").unique())

    return {cat: i for i, cat in enumerate(cats)}


def encode_with_map(series: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    """Return integer codes for a series using mapping."""
    s = series.astype("string")
    arr = s.map(mapping)
    return arr.astype(int).to_numpy()


def build_train_indices(df: pd.DataFrame) -> dict[str, Any]:
    """Return encoded categorical indices and mapping dicts for training set."""
    horse_map = make_train_map(df["horse_name"])
    jock_map = make_train_map(df["jockey_name"])
    race_map = make_train_map(df["race_id"])
    course_map = make_train_map(df["racecourse"])
    weather_map = make_train_map(df["weather_category"])
    surf_map = make_train_map(df["current_mark_surface"])
    class_map = make_train_map(df["class"].astype(str))

    return {
        "horse_idx": encode_with_map(df["horse_name"], horse_map),
        "jock_idx": encode_with_map(df["jockey_name"], jock_map),
        "race_idx": encode_with_map(df["race_id"], race_map),
        "course_idx": encode_with_map(df["racecourse"], course_map),
        "weather_idx": encode_with_map(df["weather_category"], weather_map),
        "surf_idx": encode_with_map(df["current_mark_surface"], surf_map),
        "class_idx": encode_with_map(df["class"].astype(str), class_map),
        "maps": {
            "horse": horse_map,
            "jock": jock_map,
            "race": race_map,
            "course": course_map,
            "weather": weather_map,
            "surf": surf_map,
            "class": class_map,
        },
        "n_horse": len(horse_map),
        "n_jock": len(jock_map),
        "n_race": len(race_map),
        "n_course": len(course_map),
        "n_weather": len(weather_map),
        "n_surf": len(surf_map),
        "n_class": len(class_map),
    }


def build_test_indices(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    """Return encoded categorical indices for test set using training maps."""
    return {
        "horse_idx": encode_with_map(df["horse_name"], maps["horse"]),
        "jock_idx": encode_with_map(df["jockey_name"], maps["jock"]),
        "race_idx": encode_with_map(df["race_id"], maps["race"]),
        "course_idx": encode_with_map(df["racecourse"], maps["course"]),
        "weather_idx": encode_with_map(df["weather_category"], maps["weather"]),
        "surf_idx": encode_with_map(df["current_mark_surface"], maps["surf"]),
        "class_idx": encode_with_map(df["class"].astype(str), maps["class"]),
    }


num_cols = [
    "going_stick",
    "soil_moisture_pct",
    "draw_number",
    "handicap_ran_off",
    "current_mark",
]


def sum_to_zero_noncentered(name: str, n: int, sigma_prior: float = 0.5) -> pm.Deterministic:
    """Return sum-to-zero categorical effect with non-centred parameterisation."""
    raw = pm.Normal(f"{name}_raw", 0.0, 1.0, shape=n)
    tau = pm.HalfNormal(f"sigma_{name}", sigma_prior)
    eff = (raw - pt.mean(raw)) * tau
    return pm.Deterministic(f"{name}_eff", eff)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, filter, impute, and split races into train and test sets."""
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

    flat_slim = (
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
    ).assign(won=lambda df: df.groupby("race_id")["speed"].transform(lambda x: x == x.max()))

    # Columns we're happy to impute
    impute_cols = [
        "going_stick",
        "soil_moisture_pct",
        "draw_number",
        "current_mark",
    ]

    # Median imputation inplace
    for col in impute_cols:
        if col in flat_slim.columns:
            median_val = flat_slim[col].median()
            flat_slim[col] = flat_slim[col].fillna(median_val)

    fixture_ids = sorted(flat_slim["fixture_index"].unique())
    cutoff = int(len(fixture_ids) * 0.75)
    train_ids = set(fixture_ids[:cutoff])
    test_ids = set(fixture_ids[cutoff:])

    train_df = flat_slim[flat_slim["fixture_index"].isin(train_ids)].copy()
    test_df = flat_slim[flat_slim["fixture_index"].isin(test_ids)].copy()

    return train_df, test_df


# %%


def main() -> None:  # noqa: PLR0915
    """Training Function."""
    train_df, test_df = prepare_data()
    logger.info("Training new model...")

    train_idx = build_train_indices(train_df)
    num_data = train_df[num_cols].to_numpy()
    y_train = train_df["log_speed"].to_numpy()

    with pm.Model() as model:
        y = pm.Data("y", y_train)
        z_data = pm.Data("z_data", num_data)
        horse_idx = pm.Data("horse_idx", train_idx["horse_idx"])
        jock_idx = pm.Data("jock_idx", train_idx["jock_idx"])
        race_idx = pm.Data("race_idx", train_idx["race_idx"])
        course_idx = pm.Data("course_idx", train_idx["course_idx"])
        weather_idx = pm.Data("weather_idx", train_idx["weather_idx"])
        surf_idx = pm.Data("surf_idx", train_idx["surf_idx"])
        class_idx = pm.Data("class_idx", train_idx["class_idx"])
        intercept = pm.Normal("intercept", 0.0, 0.5)
        beta = pm.Normal("beta", 0.0, 0.5, shape=num_data.shape[1])
        eff_horse = sum_to_zero_noncentered("horse", train_idx["n_horse"], sigma_prior=1.0)
        eff_jock = sum_to_zero_noncentered("jock", train_idx["n_jock"], sigma_prior=1.0)
        eff_race = sum_to_zero_noncentered("race", train_idx["n_race"], sigma_prior=1.0)
        eff_course = sum_to_zero_noncentered("course", train_idx["n_course"])
        eff_weather = sum_to_zero_noncentered("weather", train_idx["n_weather"])
        eff_surf = sum_to_zero_noncentered("surf", train_idx["n_surf"])
        eff_class = sum_to_zero_noncentered("class", train_idx["n_class"])
        mu = (
            intercept
            + pt.dot(z_data, beta)
            + eff_horse[horse_idx]
            + eff_jock[jock_idx]
            + eff_race[race_idx]
            + eff_course[course_idx]
            + eff_weather[weather_idx]
            + eff_surf[surf_idx]
            + eff_class[class_idx]
        )
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        pm.Deterministic("mu", mu)
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.9,
            random_seed=451,
            return_inferencedata=True,
        )

    logger.info(az.summary(idata, var_names=["intercept", "sigma"]))

    az.to_netcdf(idata, model_dir / "model_fit.nc")

    test_idx = build_test_indices(test_df, train_idx["maps"])
    num_test = test_df[num_cols].to_numpy()

    with model:
        pm.set_data(
            {
                "z_data": num_test,
                "horse_idx": test_idx["horse_idx"],
                "jock_idx": test_idx["jock_idx"],
                "race_idx": test_idx["race_idx"],
                "course_idx": test_idx["course_idx"],
                "weather_idx": test_idx["weather_idx"],
                "surf_idx": test_idx["surf_idx"],
                "class_idx": test_idx["class_idx"],
            },
        )
        mu_pred = pm.sample_posterior_predictive(idata, var_names=["mu"], predictions=True)

    preds = np.exp(mu_pred.predictions["mu"].mean(("chain", "draw")).values)

    pred_df = (
        test_df.assign(pred_speed=preds)
        .assign(
            pred_rank=lambda df: df.groupby("race_id")["pred_speed"]
            .rank(ascending=False, method="first")
            .astype(int),
        )
        .assign(pred_winner=lambda df: df["pred_rank"].eq(1))
        .assign(actual_winner=lambda df: df["won"].astype(bool))
        .assign(decimal_odds=lambda df: 1.0 / df["starting_prob"].replace(0, np.nan))
        .assign(bet_return=lambda df: np.where(df["actual_winner"], df["decimal_odds"], 0.0))
    )

    out_dir = OUTPUT_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    logger.info("Saved predictions to {}", out_dir / "predictions.csv")

    winners = pred_df.loc[pred_df.groupby("race_id")["pred_speed"].idxmax()]
    actual_winners = pred_df.loc[pred_df.groupby("race_id")["speed"].idxmax()]
    acc = accuracy_score(actual_winners["horse_name"].to_numpy(), winners["horse_name"].to_numpy())
    logger.info("Winner prediction accuracy: {:.2%}", acc)

    top3_acc = float(
        np.mean(
            [
                grp.loc[grp["speed"].idxmax()]["horse_name"]
                in grp.sort_values("pred_speed", ascending=False)["horse_name"].head(3).to_numpy()
                for _, grp in pred_df.groupby("race_id")
            ],
        ),
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

    total_stakes = len(summary)  # 1 unit bet per race

    total_returns = summary["pred_winner_return"].sum()
    roi = (total_returns - total_stakes) / total_stakes

    logger.info("Total ROI from betting 1 unit per race: {:.2%}", roi)


# %%
if __name__ == "__main__":
    main()
# %%
