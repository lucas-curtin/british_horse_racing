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
CUTOFF_TRAIN = pd.Timestamp("2025-06-24")  # set this to your desired manual split date


def run_bayesian_model():
    """
    Train a Bayesian hierarchical model using data up to CUTOFF_TRAIN as training,
    then evaluate on the following date. Random effects include race, weather,
    jockey, horse, course, trainer, owner, and licence.
    Raw features (excluding finish_time_sec and starting_prob) are used;
    cumulative stats are stored separately.
    Returns the inference trace and the full dataset with model probabilities.
    """
    # Load data
    data = pd.read_csv(
        RESULTS_CSV,
        index_col="race_date",
        parse_dates=["race_date", "foaled"],
    )
    data["won"] = data["won"].astype(int)

    # Flags
    data["is_withdrawn"] = data["WD"].astype(int)
    data["is_non_runner"] = data["NR"].astype(int)

    # Deduplicate rows (exclude index)
    data = data.drop_duplicates(subset=["fixture_index", "race_index", "horse_name"])

    # Race ID for grouping
    data["race_id"] = data["fixture_index"].astype(str) + "_" + data["race_index"].astype(str)

    # Train/test split using manual cutoff
    is_train = data.index <= CUTOFF_TRAIN
    test_date = CUTOFF_TRAIN + pd.Timedelta(days=1)
    if test_date not in data.index.unique():
        raise ValueError(f"Test date {test_date.date()} not in dataset index.")

    # Feature definitions
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
    raw_cols = [
        "race_distance_m",
        "draw_number",
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

    # Prepare and scale features
    X_raw = data[raw_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_raw[is_train])
    X_scaled = scaler.transform(X_raw)
    y = data["won"].values

    # Build Bayesian model
    with pm.Model() as _:

        def var_intercept(name, sigma_name, idx, n):
            sigma = pm.HalfNormal(sigma_name, sigma=1)
            offset = pm.Normal(f"offset_{name}", mu=0, sigma=1, shape=n)
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

        global_intercept = pm.Normal("global_intercept", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_scaled.shape[1])

        logits = global_intercept
        for key, idx in [
            ("race", race_idx),
            ("weather", weather_idx),
            ("jockey", jockey_idx),
            ("horse", horse_idx),
            ("course", course_idx),
            ("trainer", trainer_idx),
            ("owner", owner_idx),
            ("licence", licence_idx),
        ]:
            logits += intercepts[key][idx]
        logits += pm.math.dot(X_scaled, betas)

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

    # Attach predictions and cumulative stats
    mean_probs = trace.posterior["win_probability"].mean(dim=("chain", "draw")).values
    data = data.assign(
        model_prob=mean_probs, cumulative_stats=data[cumulative_cols].to_dict(orient="records")
    )
    return trace, data, test_date


def evaluate_model(trace, data, test_date) -> None:
    # Evaluate on specified test_date
    test = data.loc[data.index == test_date]
    runners = test.loc[~(test["WD"] | test["NR"])]

    # Predictions per race
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

    # Metrics
    probs, y_true = runners["model_prob"].values, runners["won"].values
    ll = log_loss(y_true, np.vstack([1 - probs, probs]).T)
    bs = brier_score_loss(y_true, probs)
    logger.info(f"Held-out log loss: {ll:.4f}")
    logger.info(f"Held-out Brier score: {bs:.4f}")

    # Backtest
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
    lines = [f"Evaluation for {test_date.date()}", ""]
    lines.append("=== Predictions by Race ===")
    for (f, r), grp in runners.groupby(["fixture_index", "race_index"]):
        lines.append(f"Fixture {f} Race {r}:")
        lines.append("Horse | ModelProb | SPOdds")
        for _, row in grp.drop_duplicates(subset=["horse_name"]).iterrows():
            lines.append(f"  {row.horse_name} | {row.model_prob:.4f} | {row.starting_prob:.4f}")
        lines.append("")
    lines.append("=== Calibration Metrics ===")
    lines.append(f"Log loss: {ll:.4f}")
    lines.append(f"Brier score: {bs:.4f}")
    lines.append("")
    lines.append("=== Per-Race Strategy ===")
    for _, row in both.iterrows():
        lines.append(
            f"Strategy={row.strategy}, Fixture={row.fixture_index}, Race={row.race_index}, Bets={row.bets}, Profit={row.profit:.2f}, ROI={row.roi:.3f}"
        )
    lines.append("")
    lines.append("=== Aggregate Performance ===")
    for _, row in agg.iterrows():
        lines.append(
            f"{row.strategy}: Total Bets={row.total_bets}, Profit={row.total_profit:.2f}, ROI={row.ROI:.3f}"
        )
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary written to {SUMMARY_FILE}")


def simulate_strategy(runners, strategy):
    recs = []
    for (f, r), grp in runners.groupby(["fixture_index", "race_index"]):
        if strategy == "top_pick":
            pick = grp.iloc[[grp["model_prob"].values.argmax()]]
        else:
            pick = grp[grp["model_prob"] > grp["starting_prob"]]
        profit = bets = 0
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
    total_bets, total_profit = df["bets"].sum(), df["profit"].sum()
    logger.info(
        f"Strategy {strategy}: Bets={total_bets}, ROI={total_profit / (total_bets or 1):.3f}"
    )
    return df


if __name__ == "__main__":
    multiprocessing.freeze_support()
    trace, data, test_date = run_bayesian_model()
    evaluate_model(trace, data, test_date)
