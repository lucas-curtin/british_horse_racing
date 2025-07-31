import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import regex as re
from loguru import logger
from sklearn.preprocessing import StandardScaler

# --- File path constants ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_CSV = OUTPUT_DIR / "results" / "results_formatted.csv"
HORSE_JSON = OUTPUT_DIR / "horse_details.json"
JOCKEY_JSON = OUTPUT_DIR / "jockey_details.json"


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load race results from CSV and return a deep copy.
    """
    results_df = pd.read_csv(csv_path)
    return results_df.copy(deep=True)


def classify_weather(description: str) -> str:
    """
    Classify weather description into categories.
    """
    desc = description.lower()
    if re.search(r"\b(\d+\.?\d*)\s*mm\b", desc) or "rain" in desc or "drizzle" in desc:
        return "rainy"
    if "sunny" in desc:
        return "sunny"
    if "cloudy" in desc or "overcast" in desc:
        return "cloudy"
    temperature_match = re.search(r"max\s*(\d+)\s*c", desc)
    if temperature_match and int(temperature_match.group(1)) >= 25:
        return "hot"
    return "mixed"


def wt_to_kg(weight: str) -> float:
    """
    Convert weight from 'st lbs' format to kilograms.
    """
    parts = weight.replace("lbs", "").split("st")
    if len(parts) != 2:
        return np.nan
    try:
        stones = int(parts[0])
        lbs = int(parts[1].strip())
    except ValueError:
        return np.nan
    return stones * 6.35029 + lbs * 0.453592


def frac_to_prob(odds: str) -> float:
    """
    Convert fractional odds string 'A/B' to win probability.
    """
    try:
        numerator, denominator = odds.split("/")
        return 1.0 / (float(numerator) + float(denominator))
    except Exception:
        return np.nan


def load_horse_details(json_path: Path) -> pd.DataFrame:
    """
    Load and parse horse details JSON, extracting numeric ratings.
    """
    raw_list = json.loads(json_path.read_text(encoding="utf-8"))
    horses_df = pd.DataFrame(raw_list)
    rating_pattern = re.compile(
        r"(?P<rating>\d+)\s*\(Last Published:\s*(?P<date>\d{1,2} \w+ \d{4})\)"
    )

    for category in ["flat", "chase", "hurdle", "awt"]:
        horses_df[f"{category}_rating"] = np.nan
        horses_df[f"{category}_rating_date"] = pd.NaT
        for idx, entry in horses_df[category].items():
            if isinstance(entry, str):
                match = rating_pattern.search(entry)
                if match:
                    horses_df.at[idx, f"{category}_rating"] = int(match.group("rating"))
                    horses_df.at[idx, f"{category}_rating_date"] = pd.to_datetime(
                        match.group("date"), dayfirst=True
                    )

    horses_df = horses_df.rename(columns={"name": "horse_name"})
    horses_df["total_prize_money"] = (
        horses_df["total_prize_money"].str.replace("£", "").str.replace(",", "").astype(float)
    )
    horses_df["foaled"] = pd.to_datetime(horses_df["foaled"])
    horses_df["colour"] = horses_df["type"].str.rsplit(n=1).str[0]
    horses_df["gender"] = horses_df["type"].str.rsplit(n=1).str[1]
    horses_df = horses_df.drop(columns=["flat", "chase", "hurdle", "awt", "type", "sire", "dam"])
    return horses_df


def load_jockey_details(json_path: Path) -> pd.DataFrame:
    """
    Load and parse jockey details JSON, cleaning numeric fields.
    """
    raw_list = json.loads(json_path.read_text(encoding="utf-8"))
    jockey_records = []
    for record in raw_list:
        jockey_entry = {"jockey_name": record["name"]}
        for line in record.get("jockey_info", "").splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                jockey_entry[key.strip().lower().replace("/", "_").replace(" ", "_")] = val.strip()
        for line in record.get("career_info", "").splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                jockey_entry[
                    f"career_{key.strip().lower().replace(' & ', '_').replace(' ', '_')}"
                ] = val.strip()
        jockey_records.append(jockey_entry)

    jockeys_df = pd.DataFrame(jockey_records)
    jockeys_df["career_prize_money"] = (
        jockeys_df["career_prize_money"].str.replace("£", "").str.replace(",", "").astype(float)
    )
    jockeys_df["career_win_to_rides"] = (
        jockeys_df["career_win_to_rides"].str.rstrip("%").astype(float)
    )
    jockeys_df["career_rides"] = jockeys_df["career_rides"].str.replace(",", "").astype(int)
    jockeys_df["career_wins"] = jockeys_df["career_wins"].str.replace(",", "").astype(int)
    jockeys_df["days_since_last_win"] = (
        jockeys_df["days_since_last_win"].replace("-", np.nan).str.replace(",", "").astype("Int64")
    )
    jockeys_df["jockey_age"] = jockeys_df["age"].astype("Int64")
    jockeys_df["lowest_weight_ridden"] = jockeys_df["lowest_weight_ridden"].apply(wt_to_kg)
    return jockeys_df


def main() -> None:
    """
    Main flow: load, preprocess inline, build/model, sample, and log probabilities.
    """
    multiprocessing.freeze_support()

    # Load raw data
    race_results = load_data(RESULTS_CSV)
    horse_details = load_horse_details(HORSE_JSON)
    jockey_details = load_jockey_details(JOCKEY_JSON)

    # Inline preprocessing (avoid a separate function with too many I/O)
    race_results["race_date"] = pd.to_datetime(race_results["race_date"])
    race_results["weather_category"] = race_results["weather"].apply(classify_weather)

    merged_df = race_results.merge(horse_details, on="horse_name", how="left").merge(
        jockey_details, on="jockey_name", how="left"
    )
    merged_df["won"] = (merged_df["position_rank"] == 1).astype(int)
    merged_df["starting_prob"] = merged_df["sp"].apply(frac_to_prob)
    merged_df["age_in_days"] = (merged_df["race_date"] - merged_df["foaled"]).dt.days

    # Factorize categorical indices
    race_idx, race_labels = pd.factorize(merged_df["race_index"])
    weather_idx, weather_labels = pd.factorize(merged_df["weather_category"])
    jockey_idx, jockey_labels = pd.factorize(merged_df["jockey_name"])
    horse_idx, horse_labels = pd.factorize(merged_df["horse_name"])

    # Continuous features
    feature_columns = [
        "race_distance_m",
        "draw_number",
        "starting_prob",
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
        "age_in_days",
    ]
    continuous_array = merged_df[feature_columns].fillna(0.0).values
    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(continuous_array)

    # Sizes for hierarchical effects
    effect_sizes = {
        "races": len(race_labels),
        "weather_types": len(weather_labels),
        "jockeys": len(jockey_labels),
        "horses": len(horse_labels),
    }
    outcomes = merged_df["won"].values

    # Build and sample Bayesian model
    with pm.Model() as _:
        sigma_r = pm.HalfNormal("sigma_race", sigma=1.0)
        race_effect = pm.Normal(
            "race_intercept", mu=0.0, sigma=sigma_r, shape=effect_sizes["races"]
        )
        global_bias = pm.Normal("global_intercept", mu=0.0, sigma=1.0)
        beta_cont = pm.Normal(
            "beta_continuous", mu=0.0, sigma=1.0, shape=scaled_continuous.shape[1]
        )
        beta_weather = pm.Normal(
            "beta_weather", mu=0.0, sigma=1.0, shape=effect_sizes["weather_types"]
        )
        sigma_j = pm.HalfNormal("sigma_jockey", sigma=1.0)
        sigma_h = pm.HalfNormal("sigma_horse", sigma=1.0)
        jockey_effect = pm.Normal(
            "jockey_intercept", mu=0.0, sigma=sigma_j, shape=effect_sizes["jockeys"]
        )
        horse_effect = pm.Normal(
            "horse_intercept", mu=0.0, sigma=sigma_h, shape=effect_sizes["horses"]
        )

        linear_term = (
            global_bias
            + race_effect[race_idx]
            + jockey_effect[jockey_idx]
            + horse_effect[horse_idx]
            + pm.math.dot(scaled_continuous, beta_cont)
            + beta_weather[weather_idx]
        )
        win_prob = pm.math.sigmoid(linear_term)
        pm.Deterministic("win_probability", win_prob)
        pm.Bernoulli("observed", p=win_prob, observed=outcomes)

        idata = pm.sample(draws=1000, tune=1000, target_accept=0.9, return_inferencedata=True)
        # Include deterministic in posterior predictive
        posterior_checks = pm.sample_posterior_predictive(idata, var_names=["win_probability"])

    # Extract and log mean win probabilities
    pp = posterior_checks.posterior_predictive["win_probability"]
    # Combine chain and draw into a single sample dimension
    pp_stack = pp.stack(sample=("chain", "draw"))
    mean_probabilities = pp_stack.mean(dim="sample").values
    for i, prob in enumerate(mean_probabilities[:10], start=1):
        logger.info(f"Runner {i}: estimated win probability = {prob:.3f}")


if __name__ == "__main__":
    main()
