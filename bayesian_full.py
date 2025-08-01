import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from loguru import logger
from sklearn.preprocessing import StandardScaler

# --- File path constants ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_CSV = OUTPUT_DIR / "merged_results.csv"
FIXTURE_CSV = OUTPUT_DIR / "fixture_results.csv"
BET_FILE = OUTPUT_DIR / "bet_summary.txt"

# Feature lists
RAW_FEATURES = [
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
RE_KEYS = [
    "race_id",
    "weather_category",
    "jockey_name",
    "horse_name",
    "racecourse",
    "trainer_name",
    "owner_name",
    "licence_permit_type",
]


def train_on_all():
    # Load full merged training data
    data = pd.read_csv(TRAIN_CSV, index_col="race_date", parse_dates=["race_date", "foaled"])
    data["won"] = data["won"].astype(int)
    data["is_withdrawn"] = data["WD"].astype(int)
    data["is_non_runner"] = data["NR"].astype(int)
    data = data.drop_duplicates(
        subset=["fixture_index", "race_index", "runner_index", "horse_name"]
    )
    # Create race_id grouping
    data["race_id"] = data["fixture_index"].astype(str) + "_" + data["race_index"].astype(str)

    # Factorize random effects
    idx = {}
    labels = {}
    for key in RE_KEYS:
        codes, labs = pd.factorize(data[key].fillna("Unknown").astype(str))
        idx[key] = codes
        labels[key] = labs

    # Prepare features and response
    X_raw = data[RAW_FEATURES].fillna(0.0)
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)
    y = data["won"].values

    # Build & sample hierarchical model
    with pm.Model() as model:
        intercepts = {}
        for key in RE_KEYS:
            n = int(idx[key].max()) + 1
            sigma = pm.HalfNormal(f"sigma_{key}", sigma=1)
            offset = pm.Normal(f"offset_{key}", mu=0, sigma=1, shape=(n,))
            intercepts[key] = sigma * offset

        global_int = pm.Normal("global_intercept", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=(X_scaled.shape[1],))

        logits = global_int
        for key in RE_KEYS:
            logits += intercepts[key][idx[key]]
        logits += pm.math.dot(X_scaled, betas)

        win_prob = pm.math.sigmoid(logits)
        pm.Bernoulli("obs", p=win_prob, observed=y)
        trace = pm.sample(
            draws=1000,
            tune=1000,
            target_accept=0.995,
            max_treedepth=15,
            cores=4,
            return_inferencedata=True,
        )

    return trace, scaler, idx, labels


def predict_and_summarize(trace, scaler, idx, labels):
    # Load future fixtures
    fdf = pd.read_csv(FIXTURE_CSV, index_col="race_date", parse_dates=["race_date", "foaled"])
    fdf["is_withdrawn"] = fdf["WD"].astype(int)
    fdf["is_non_runner"] = fdf["NR"].astype(int)
    fdf["race_id"] = fdf["fixture_index"].astype(str) + "_" + fdf["race_index"].astype(str)

    # Scale features
    Xf = fdf[RAW_FEATURES].fillna(0.0)
    Xf_scaled = scaler.transform(Xf)

    # Factorize with training categories
    idx_f = {}
    for key in RE_KEYS:
        vals = fdf[key].fillna("Unknown").astype(str)
        codes = pd.Categorical(vals, categories=labels[key]).codes
        idx_f[key] = np.where(codes < 0, labels[key].size - 1, codes)

    # Compute posterior means
    post = trace.posterior
    mean_betas = post["betas"].stack(sample=("chain", "draw")).mean(dim="sample").values
    mean_gi = post["global_intercept"].stack(sample=("chain", "draw")).mean(dim="sample").values
    re_means = {
        key: post[f"offset_{key}"].stack(sample=("chain", "draw")).mean(dim="sample").values
        for key in RE_KEYS
    }

    # Calculate win probabilities
    logits = mean_gi + Xf_scaled.dot(mean_betas)
    for key in RE_KEYS:
        logits += re_means[key][idx_f[key]]
    fdf["model_prob"] = 1 / (1 + np.exp(-logits))

    # Generate bet summary
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(BET_FILE, "w") as bf:
        for (f, r), grp in fdf.groupby(["fixture_index", "race_index"]):
            bf.write(f"Fixture {f} Race {r}:\n")
            preds = (
                grp[["horse_name", "model_prob"]]
                .rename(columns={"horse_name": "Horse", "model_prob": "Prob"})
                .sort_values("Prob", ascending=False)
            )
            for _, row in preds.iterrows():
                bf.write(f"  {row.Horse}: {row.Prob:.2%}\n")
            top = preds.iloc[0]
            bf.write(f"Top pick: {top.Horse} ({top.Prob:.2%})\n\n")
    logger.info(f"Bet summary written to {BET_FILE}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    trace, scaler, idx, labels = train_on_all()
    predict_and_summarize(trace, scaler, idx, labels)
