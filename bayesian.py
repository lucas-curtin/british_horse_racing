import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from loguru import logger
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

# --- File path constants ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_CSV = OUTPUT_DIR / "merged_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "results_summary.txt"

# --- Manual cutoff date for train/test split ---
CUTOFF_TRAIN = pd.Timestamp("2025-07-24")  # adjust as needed


def run_bayesian_model():
    # Load merged results (with all new features)
    data = pd.read_csv(
        RESULTS_CSV,
        index_col="race_date",
        parse_dates=["race_date", "foaled"],
    )
    data["won"] = data["won"].astype(int)

    # Withdrawn/non-runner flags
    data["is_withdrawn"] = data["WD"].astype(int)
    data["is_non_runner"] = data["NR"].astype(int)

    # Deduplicate: include runner_index to avoid collapsing distinct races
    data = data.drop_duplicates(
        subset=["fixture_index", "race_index", "runner_index", "horse_name"]
    )

    # Grouping
    data["race_id"] = data["fixture_index"].astype(str) + "_" + data["race_index"].astype(str)

    # Train/test masks (positional indexing will be used later)
    is_train = data.index <= CUTOFF_TRAIN
    is_test = data.index > CUTOFF_TRAIN

    # Cumulative stats (excluded from raw_cols)
    cumulative_cols = [
        "total_runs",
        "total_wins",
        "total_places",
        "total_prize_money",
        "career_wins",
        "career_rides",
        "career_win_to_rides",
        "career_group_listed_wins",
        "career_prize_money",
    ]

    # Raw features to include
    raw_cols = [
        "race_distance_m",
        "draw_number",
        "handicap_ran_off",
        "bha_performance_figure",
        "current_mark",
        "flat_rating",
        "chase_rating",
        "hurdle_rating",
        "awt_rating",
        "days_since_last_win",
        "lowest_weight_ridden",
        "jockey_age",
        "race_age",
        "is_withdrawn",
        "is_non_runner",
        "going_stick",
        "soil_moisture_pct",
    ]

    # Factorize random effects
    race_idx, _ = pd.factorize(data["race_id"])
    weather_idx, weather_labels = pd.factorize(data["weather_category"])
    jockey_idx, jockey_labels = pd.factorize(data["jockey_name"])
    horse_idx, horse_labels = pd.factorize(data["horse_name"])
    course_idx, course_labels = pd.factorize(data["racecourse"])
    trainer_idx, trainer_labels = pd.factorize(data["trainer_name"])
    owner_idx, owner_labels = pd.factorize(data["owner_name"])
    licence_idx, licence_labels = pd.factorize(data["licence_permit_type"].fillna("Unknown"))

    # Prepare & scale raw features
    X_raw = data[raw_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_raw[is_train])
    X_scaled = scaler.transform(X_raw)
    y = data["won"].values

    # Build Bayesian hierarchical model
    with pm.Model() as model:

        def var_intercept(name, sigma_name, idx, n_levels):
            sigma = pm.HalfNormal(sigma_name, sigma=1)
            offset = pm.Normal(f"offset_{name}", mu=0, sigma=1, shape=n_levels)
            return pm.Deterministic(f"intercept_{name}", sigma * offset)

        intercepts = {
            "race": var_intercept("race", "sigma_race", race_idx, len(np.unique(race_idx))),
            "weather": var_intercept("weather", "sigma_weather", weather_idx, len(weather_labels)),
            "jockey": var_intercept("jockey", "sigma_jockey", jockey_idx, len(jockey_labels)),
            "horse": var_intercept("horse", "sigma_horse", horse_idx, len(horse_labels)),
            "course": var_intercept("course", "sigma_course", course_idx, len(course_labels)),
            "trainer": var_intercept("trainer", "sigma_trainer", trainer_idx, len(trainer_labels)),
            "owner": var_intercept("owner", "sigma_owner", owner_idx, len(owner_labels)),
            "licence": var_intercept("licence", "sigma_licence", licence_idx, len(licence_labels)),
        }

        # Fixed effects
        global_intercept = pm.Normal("global_intercept", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_scaled.shape[1])

        # Assemble logit
        logits = global_intercept
        for name, idx in [
            ("race", race_idx),
            ("weather", weather_idx),
            ("jockey", jockey_idx),
            ("horse", horse_idx),
            ("course", course_idx),
            ("trainer", trainer_idx),
            ("owner", owner_idx),
            ("licence", licence_idx),
        ]:
            logits += intercepts[name][idx]
        logits += pm.math.dot(X_scaled, betas)

        # Likelihood
        win_prob = pm.Deterministic("win_probability", pm.math.sigmoid(logits))
        pm.Bernoulli("obs", p=win_prob[is_train], observed=y[is_train])

        trace = pm.sample(
            draws=1000,
            tune=1000,
            target_accept=0.995,
            max_treedepth=15,
            cores=4,
            return_inferencedata=True,
        )

    # Attach model predictions & cumulative stats
    mean_probs = trace.posterior["win_probability"].mean(dim=("chain", "draw")).values
    data = data.assign(
        model_prob=mean_probs,
        cumulative_stats=data[cumulative_cols].to_dict(orient="records"),
    )

    return trace, data, is_train, is_test


def evaluate_model(trace, data, is_test) -> None:
    # Use positional indexing for test split
    test = data[is_test]
    runners = test[~(test["WD"] | test["NR"])]

    # Per-race predictions
    for (f, r), grp in runners.groupby(["fixture_index", "race_index"]):
        preds = (
            grp[["horse_name", "model_prob", "starting_prob"]]
            .rename(
                columns={
                    "horse_name": "Horse",
                    "model_prob": "ModelProb",
                    "starting_prob": "SPOdds",
                }
            )
            .drop_duplicates(subset=["Horse"])
            .sort_values("ModelProb", ascending=False)
        )
        logger.info(f"Fixture {f} Race {r} predictions:\n{preds.to_string(index=False)}")

    # Calibration metrics
    probs, y_true = runners["model_prob"].values, runners["won"].values
    ll = log_loss(y_true, np.vstack([1 - probs, probs]).T)
    bs = brier_score_loss(y_true, probs)
    logger.info(f"Held-out log loss: {ll:.4f}")
    logger.info(f"Held-out Brier score: {bs:.4f}")

    # Strategy backtest
    top = simulate_strategy(runners, "top_pick")
    val = simulate_strategy(runners, "value_bets")
    both = pd.concat([top, val], ignore_index=True)

    logger.info("\nPer-race strategy:")
    logger.info(both.to_string(index=False))

    agg = (
        both.groupby("strategy")
        .agg(total_bets=("bets", "sum"), total_profit=("profit", "sum"))
        .reset_index()
    )
    agg["ROI"] = agg["total_profit"] / agg["total_bets"].replace(0, 1)
    logger.info("\nAggregate performance:")
    logger.info(agg.to_string(index=False))

    # Write summary
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(SUMMARY_FILE, "w") as f:
        f.write(f"Calibration Log loss: {ll:.4f}\nBrier: {bs:.4f}\n")

    logger.info(f"Summary written to {SUMMARY_FILE}")


def simulate_strategy(runners, strategy):
    recs = []
    for (f, r), grp in runners.groupby(["fixture_index", "race_index"]):
        if strategy == "top_pick":
            pick = grp.iloc[[grp["model_prob"].values.argmax()]]
        else:
            pick = grp[grp["model_prob"] > grp["starting_prob"]]
        bets = profit = 0
        for _, row in pick.iterrows():
            bets += 1
            odds = (1 / row["starting_prob"]) - 1
            profit += odds if row["won"] else -1
        roi = profit / (bets or 1)
        recs.append(
            {
                "strategy": strategy,
                "fixture_index": f,
                "race_index": r,
                "bets": bets,
                "profit": profit,
                "roi": roi,
            }
        )
    df = pd.DataFrame(recs)
    logger.info(
        f"Strategy {strategy}: Bets={df.bets.sum()}, ROI={df.profit.sum() / (df.bets.sum() or 1):.3f}"
    )
    return df


if __name__ == "__main__":
    multiprocessing.freeze_support()
    trace, data, is_train, is_test = run_bayesian_model()
    evaluate_model(trace, data, is_test)
