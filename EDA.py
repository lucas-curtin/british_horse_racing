# %%
# horse_dynamic_model_pymc.py
# Pure PyMC rewrite of the dynamic ability model, with tqdm for building dynamic abilities

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import regex as re
from loguru import logger
from sklearn.preprocessing import StandardScaler

# %%
# --- Paths & Data Loading ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
results_path = OUTPUT_DIR / "results"
horse_json_path = OUTPUT_DIR / "horse_details.json"
jockey_json_path = OUTPUT_DIR / "jockey_details.json"

# Read results
results_df = pd.read_csv(results_path / "results_formatted.csv")


def classify_weather(txt):
    t = txt.lower()
    # if there's measurable rain/drizzle mention → rainy
    if re.search(r"\b(\d+\.?\d*)\s*mm\b", t) or "rain" in t or "drizzle" in t:
        return "rainy"
    # else if 'sunny' appears → sunny
    elif "sunny" in t:
        return "sunny"
    # else if cloudy or overcast → cloudy
    elif "cloudy" in t or "overcast" in t:
        return "cloudy"
    # else if mentions hot temperature above 25c → hot/dry
    match = re.search(r"max\s*(\d+)\s*c", t)
    if match and int(match.group(1)) >= 25:
        return "hot"
    # otherwise fallback
    return "mixed"


results_df["weather_type"] = results_df["weather"].apply(classify_weather)
results_df["race_date"] = pd.to_datetime(results_df["race_date"])

# %%
# Load and parse horse details
with horse_json_path.open("r", encoding="utf-8") as f:
    horse_data = json.load(f)
horse_raw = pd.DataFrame(horse_data)

rating_pattern = re.compile(r"(?P<rating>\d+)\s*\(Last Published:\s*(?P<date>\d{1,2} \w+ \d{4})\)")
for col in ["flat", "chase", "hurdle", "awt"]:
    horse_raw[f"{col}_rating"] = np.nan
    horse_raw[f"{col}_rating_date"] = pd.NaT
    for idx, cell in horse_raw[col].items():
        if isinstance(cell, str):
            m = rating_pattern.search(cell)
            if m:
                horse_raw.at[idx, f"{col}_rating"] = int(m.group("rating"))
                horse_raw.at[idx, f"{col}_rating_date"] = pd.to_datetime(
                    m.group("date"), dayfirst=True
                )

horse_df = (
    horse_raw.rename(columns={"name": "horse_name"}).assign(
        total_prize_money=lambda df: df["total_prize_money"]
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float),
        foaled=lambda df: pd.to_datetime(df["foaled"]),
        colour=lambda df: df["type"].str.rsplit(" ", n=1).str[0],
        gender=lambda df: df["type"].str.rsplit(" ", n=1).str[1],
    )
).drop(columns=["chase", "hurdle", "awt", "flat", "type", "sire", "dam"])
# %%
# Load and parse jockey details
with jockey_json_path.open("r", encoding="utf-8") as f:
    jockey_data = json.load(f)

parsed = []
for item in jockey_data:
    base = {"name": item["name"]}
    for line in item.get("jockey_info", "").splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            col = key.strip().lower().replace("/", "_").replace(" ", "_")
            base[col] = val.strip()
    for line in item.get("career_info", "").splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            col = "career_" + key.strip().lower().replace(" & ", "_").replace(" ", "_")
            base[col] = val.strip()
    parsed.append(base)

jockey_raw = pd.DataFrame(parsed).drop(columns=["associated_content"])


def wt_to_kg(s: str) -> float:
    parts = s.replace("lbs", "").split("st")
    if len(parts) != 2:
        return np.nan
    try:
        st_val = int(parts[0])
        lbs_val = int(parts[1].strip())
    except ValueError:
        return np.nan
    return st_val * 6.35029 + lbs_val * 0.453592


jockey_df = jockey_raw.assign(
    career_prize_money=lambda df: df["career_prize_money"]
    .str.replace("£", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float),
    career_win_to_rides=lambda df: df["career_win_to_rides"].str.rstrip("%").astype(float),
    career_rides=lambda df: df["career_rides"].str.replace(",", "", regex=False).astype(int),
    career_wins=lambda df: df["career_wins"].str.replace(",", "", regex=False).astype(int),
    days_since_last_win=lambda df: df["days_since_last_win"]
    .replace("-", np.nan)
    .str.replace(",", "", regex=False)
    .astype("Int64"),
    jockey_age=lambda df: df["age"].astype("Int64"),
    lowest_weight_ridden=lambda df: df["lowest_weight_ridden"].apply(wt_to_kg),
).rename(columns={"name": "jockey_name"})


# %%
# ? Feature selection
full_df = (
    results_df.merge(horse_df, on=["horse_name"], how="left")
    .merge(jockey_df, on=["jockey_name"], how="left")
    .assign(horse_age=lambda df: df["race_date"] - df["foaled"])
)[
    [
        "race_index",
        "racecourse",
        "runner_index",
        "position_rank",
        "draw_number",
        "horse_name",
        "jockey_name",
        "trainer_name",
        "owner_name",
        "race_distance_m",
        "finish_time_sec",
        "sp",
        "race_date",
        "weather_type",
        "total_runs",
        "total_wins",
        "total_places",
        "total_prize_money",
        "gender",
        "jockey_age",
        "licence_permit_type",
        "days_since_last_win",
        "lowest_weight_ridden",
        "career_wins",
        "career_rides",
        "career_win_to_rides",
        "career_group_listed_wins",
        "career_prize_money",
        "horse_age",
    ]
]


# %%

# Make a copy of the merged dataframe
df = full_df.copy()

# 1. Define binary outcome: did the horse win?
df["is_winner"] = (df["position_rank"] == 1).astype(int)


# 2. Convert starting‐price (SP) strings like "20/1" into win probabilities
def fractional_odds_to_prob(odds_str):
    """
    Parses a fractional odds string "A/B" and returns implied win probability 1/(A+B).
    If parsing fails, returns NaN.
    """
    try:
        a, b = odds_str.split("/")
        a = float(a)
        b = float(b)
        return 1.0 / (a + b)
    except Exception:
        return np.nan


df["sp_prob"] = df["sp"].apply(fractional_odds_to_prob)

# 3. Factor‐encode categorical variables: weather and race identifier
df["weather_code"], weather_categories = pd.factorize(df["weather_type"])
df["race_code"], race_categories = pd.factorize(df["race_index"])

# 4. Prepare continuous features and standardize them
# Convert horse_age (timedelta) to numeric days
df["horse_age_days"] = df["horse_age"].dt.days.astype(float)

continuous_cols = [
    "race_distance_m",
    "draw_number",
    "sp_prob",
    "total_runs",
    "total_wins",
    "total_places",
    "total_prize_money",
    "jockey_age",
    "days_since_last_win",
    "lowest_weight_ridden",
    "career_wins",
    "career_rides",
    "career_win_to_rides",
    "career_group_listed_wins",
    "career_prize_money",
    "horse_age_days",
]

scaler = StandardScaler()
X_continuous = scaler.fit_transform(df[continuous_cols].fillna(0.0))

# 5. Extract arrays for the model
race_index = df["race_code"].values
weather_index = df["weather_code"].values
y = df["is_winner"].values

number_of_races = len(race_categories)
number_of_weather_types = len(weather_categories)
number_of_features = X_continuous.shape[1]

# 6. Build and fit the hierarchical logistic regression in PyMC
with pm.Model() as horse_winner_model:
    # Hyperprior for race‐specific intercept variability
    sigma_race = pm.HalfNormal("sigma_race", sigma=1.0)

    # Race‐level intercepts
    race_intercept = pm.Normal("race_intercept", mu=0.0, sigma=sigma_race, shape=number_of_races)

    # Global intercept
    global_intercept = pm.Normal("global_intercept", mu=0.0, sigma=1.0)

    # Coefficients for continuous predictors
    coeff_cont = pm.Normal("coeff_continuous", mu=0.0, sigma=1.0, shape=number_of_features)

    # Coefficients for weather categories
    coeff_weather = pm.Normal("coeff_weather", mu=0.0, sigma=1.0, shape=number_of_weather_types)

    # Linear predictor for each runner
    linear_term = (
        global_intercept
        + race_intercept[race_index]
        + pm.math.dot(X_continuous, coeff_cont)
        + coeff_weather[weather_index]
    )

    # Convert linear predictor to probability via logistic function
    win_probability = pm.math.sigmoid(linear_term)

    # Likelihood: observed win/loss outcome
    observed_outcome = pm.Bernoulli("observed_outcome", p=win_probability, observed=y)

    # Sample from the posterior
    trace = pm.sample(
        draws=1000, tune=1000, target_accept=0.9, return_inferencedata=True, progressbar=True
    )

    # Posterior predictive sampling to get win‐probability estimates
    posterior_predictions = pm.sample_posterior_predictive(trace, var_names=["win_probability"])

# 7. Summarize estimated win probabilities for the first 10 runners
mean_probabilities = posterior_predictions["win_probability"].mean(axis=0)
for idx in range(10):
    logger.info(f"Runner {idx + 1}: estimated win probability = {mean_probabilities[idx]:.3f}")
# %%
