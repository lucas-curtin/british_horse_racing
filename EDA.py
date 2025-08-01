# %%
"""EDA Script."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re

# --- File path constants ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
HORSE_JSON = OUTPUT_DIR / "horse_details.json"
JOCKEY_JSON = OUTPUT_DIR / "jockey_details.json"

MILE_M = 1609.34  # metres per mile
FURLONG_M = 201.168  # metres per furlong
YARD_M = 0.9144  # metres per yard

# %%
# ? Helper Funcs


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


def parse_timedelta(x: str) -> pd.Timedelta:
    """Timedelta converter."""
    if not x or pd.isna(x):
        return pd.NaT
    m = re.match(r"(\d+)m\s*([\d\.]+)s", x)
    if not m:
        return pd.NaT
    minutes, seconds = int(m.group(1)), float(m.group(2))
    return pd.to_timedelta(minutes * 60 + seconds, unit="s")


def parse_distance_to_metres(s: str) -> float:
    """Imperial to metric."""
    if pd.isna(s):
        return np.nan
    miles = furlongs = yards = 0
    m = re.search(r"(\d+)\s*m\b", s)
    if m:
        miles = int(m.group(1))
    f = re.search(r"(\d+)\s*f\b", s)
    if f:
        furlongs = int(f.group(1))
    y = re.search(r"(\d+)\s*y\b", s)
    if y:
        yards = int(y.group(1))
    total_yards = miles * 1760 + furlongs * 220 + yards
    return total_yards * 0.9144


# %%
# ? Race Info
# Load raw data
# Constants for conversion


records = []
for file in sorted(RESULTS_DIR.glob("*.json")):
    ym = file.stem  # e.g. "2025_06"
    with file.open() as f:
        fixtures = json.load(f)
    for fixture in fixtures:
        for race in fixture["races"]:
            for runner in race["runners"]:
                # Base record with raw strings defaulting to ""
                rec = {
                    "year_month": ym,
                    "fixture_index": fixture["fixture_index"],
                    "fixture_date": fixture["date"],
                    "fixture_year": fixture["year"],
                    "race_index": race["race_index"],
                    "racecourse": fixture["racecourse"],
                    "going": fixture["going"],
                    "weather": fixture["weather"],
                    # "other_text": fixture["other_text"],
                    "race_time_raw": race.get("time", ""),
                    "race_name_raw": race.get("name", ""),
                    "race_distance_raw": race.get("distance", ""),
                    "winner_info_raw": race.get("winner_info", ""),
                    "runner_index": runner.get("runner_index", ""),
                    "position_raw": runner.get("position", ""),
                    "horse_jockey_raw": runner.get("horse_jockey", ""),
                    "trainer_owner_raw": runner.get("trainer_owner", ""),
                    "distance_time_raw": runner.get("distance_time", ""),
                    "sp_raw": runner.get("sp", ""),
                }

                # Parse position into rank and extra
                parts = rec["position_raw"].split("\n") if rec["position_raw"] else []
                rec["position_rank"] = re.sub(r"(?:st|nd|rd|th)$", "", parts[0]) if parts else ""
                rec["position_extra"] = parts[1] if len(parts) > 1 else ""
                # Extract draw stall number only
                m_draw = re.search(r"D:(\d+)", rec["position_extra"])
                rec["draw_number"] = int(m_draw.group(1)) if m_draw else np.nan

                # Parse horse & jockey names
                hj_lines = rec["horse_jockey_raw"].split("\n")
                rec["horse_name"] = hj_lines[0] if hj_lines else ""
                rec["jockey_name"] = hj_lines[1] if len(hj_lines) > 1 else ""
                for line in hj_lines[2:]:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        col = key.strip().lower().replace(" ", "_")
                        rec[f"{col}_raw"] = val.strip()

                # Parse trainer & owner names
                to_lines = rec["trainer_owner_raw"].split("\n")
                rec["trainer_name"] = to_lines[0] if to_lines else ""
                rec["owner_name"] = to_lines[1] if len(to_lines) > 1 else ""

                # Race-level distance: prefer normalized in parentheses
                rd = rec.pop("race_distance_raw", "")
                rd_val = rd.split("\n")[1].strip("()") if "\n" in rd else rd
                rec["race_distance"] = rd_val
                # Compute race distance in metres
                m_m = re.search(r"(\d+)m", rd_val)
                f_m = re.search(r"(\d+)f", rd_val)
                y_m = re.search(r"(\d+)y", rd_val)
                miles = int(m_m.group(1)) if m_m else 0
                furlongs = int(f_m.group(1)) if f_m else 0
                yards = int(y_m.group(1)) if y_m else 0
                rec["race_distance_m"] = miles * MILE_M + furlongs * FURLONG_M + yards * YARD_M

                # Runner-level raw parse for distance/time
                dt = rec.pop("distance_time_raw", "")
                dt_lines = dt.split("\n") if dt else []
                if len(dt_lines) == 2:
                    raw_dist, raw_time = dt_lines
                elif len(dt_lines) == 1:
                    text = dt_lines[0]
                    if re.search(r"\d+m\s*[\d\.]+s", text):
                        raw_dist, raw_time = "", text
                    else:
                        raw_dist, raw_time = text, ""
                else:
                    raw_dist, raw_time = "", ""

                rec["distance_raw"] = raw_dist
                rec["finish_time_raw"] = raw_time

                # Finish time into seconds
                tm = re.match(r"(\d+)m\s*([\d\.]+)s", raw_time)
                rec["finish_time_sec"] = (
                    int(tm.group(1)) * 60 + float(tm.group(2)) if tm else np.nan
                )

                # SP
                rec["sp"] = rec.pop("sp_raw", "")

                records.append(rec)

# Build DataFrame and clean up raw columns
results_raw = pd.DataFrame(records)

race_results = (
    results_raw.astype({"fixture_date": "str", "fixture_year": "str"}).assign(
        race_date=lambda df: pd.to_datetime(df["fixture_date"] + " " + df["fixture_year"]),
        starting_prob=lambda df: df["sp"].apply(frac_to_prob),
        won=lambda df: df["position_rank"] == "1",
        weather_category=lambda df: df["weather"].apply(classify_weather),
        DNF=lambda df: df["position_rank"] == "DNF",
        WD=lambda df: df["position_rank"] == "WD",
        NR=lambda df: df["position_rank"] == "NR",
        draw_number=lambda df: df["draw_number"].fillna(0),
    )
).drop(
    columns=[
        "position_extra",
        "race_distance",
        "sp",
        "year_month",
        "fixture_date",
        "fixture_year",
        "weather",
        "going",
    ]
)


# %%
# ? Horse Info
horse_list = json.loads(HORSE_JSON.read_text(encoding="utf-8"))
horse_df = pd.DataFrame(horse_list)
rating_pattern = re.compile(r"(?P<rating>\d+)\s*\(Last Published:\s*(?P<date>\d{1,2} \w+ \d{4})\)")

for category in ["flat", "chase", "hurdle", "awt"]:
    horse_df[f"{category}_rating"] = np.nan
    horse_df[f"{category}_rating_date"] = pd.NaT
    for idx, entry in horse_df[category].items():
        if isinstance(entry, str):
            match = rating_pattern.search(entry)
            if match:
                horse_df.at[idx, f"{category}_rating"] = int(match.group("rating"))
                horse_df.at[idx, f"{category}_rating_date"] = pd.to_datetime(
                    match.group("date"), dayfirst=True
                )

horse_df = (
    horse_df.rename(columns={"name": "horse_name"})
    .assign(
        total_prize_money=lambda df: df["total_prize_money"]
        .str.replace("£", "")
        .str.replace(",", "")
        .astype(float),
        foaled=lambda df: pd.to_datetime(df["foaled"]),
        colour=lambda df: df["type"].str.rsplit(n=1).str[0],
        gender=lambda df: df["type"].str.rsplit(n=1).str[1],
    )
    .drop(
        columns=[
            "flat",
            "chase",
            "hurdle",
            "awt",
            "type",
            "sire",
            "dam",
            "birth_year",
            "owner",
            "trainer",
            "deceased",
            "colour",
            "flat_rating_date",
            "chase_rating_date",
            "hurdle_rating_date",
            "awt_rating_date",
        ]
    )
)


# %%
# ? Jockey Info


jockey_list = json.loads(JOCKEY_JSON.read_text(encoding="utf-8"))
jockey_records = []
for record in jockey_list:
    jockey_entry = {"jockey_name": record["name"]}
    for line in record.get("jockey_info", "").splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            jockey_entry[key.strip().lower().replace("/", "_").replace(" ", "_")] = val.strip()
    for line in record.get("career_info", "").splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            jockey_entry[f"career_{key.strip().lower().replace(' & ', '_').replace(' ', '_')}"] = (
                val.strip()
            )
    jockey_records.append(jockey_entry)

jockey_df = (
    pd.DataFrame(jockey_records)
    .assign(
        career_prize_money=lambda df: (
            df["career_prize_money"].str.replace("£", "").str.replace(",", "").astype(float)
        ),
        career_win_to_rides=lambda df: (df["career_win_to_rides"].str.rstrip("%").astype(float)),
        career_rides=lambda df: (df["career_rides"].str.replace(",", "").astype(int)),
        career_wins=lambda df: (df["career_wins"].str.replace(",", "").astype(int)),
        days_since_last_win=lambda df: (
            df["days_since_last_win"].replace("-", np.nan).str.replace(",", "").astype("Int64")
        ),
        jockey_age=lambda df: df["age"].astype("Int64"),
        lowest_weight_ridden=lambda df: df["lowest_weight_ridden"].apply(wt_to_kg),
    )
    .drop(columns=["associated_content", "age"])
)

# %%
# ? Combining dataframes
merged_df = race_results.merge(horse_df, on="horse_name", how="left").merge(
    jockey_df, on="jockey_name", how="left"
)

merged_df["race_age"] = (merged_df["race_date"] - merged_df["foaled"]).dt.days


raw_cols = [c for c in merged_df.columns if c.endswith("_raw")]

model_input = merged_df.drop(columns=raw_cols).set_index("race_date")

model_input.to_csv(OUTPUT_DIR / "merged_results.csv")
# %%
# ? Fixture Loader

FIXTURES_JSON = OUTPUT_DIR / "fixtures.json"
with open(FIXTURES_JSON, "r", encoding="utf-8") as f:
    fixtures = json.load(f)

records = []
for fidx, fixture in enumerate(fixtures, start=1):
    # split type_date into day and race type
    td = fixture.get("type_date", "").split("\n")
    fixture_day = td[0]
    fixture_type = td[1] if len(td) > 1 else ""

    # split first_race into time and session
    fr = fixture.get("first_race", "").split("\n")
    first_race_time = fr[0]
    session = fr[1] if len(fr) > 1 else ""

    going_raw = fixture.get("going", "")
    weather_raw = fixture.get("weather", "")
    other_raw = fixture.get("other", "")

    for ridx, race in enumerate(fixture.get("races", []), start=1):
        # normalize race distance to the parenthetical value if present
        rd = race.get("distance", "")
        race_distance = rd.split("\n")[1].strip("()") if "\n" in rd else rd

        for jidx, runner in enumerate(race.get("runners", []), start=1):
            # split horse/jockey
            hj = runner.get("horse_jockey", "").split("\n")
            horse_name = hj[0] if hj else ""
            jockey_name = hj[1] if len(hj) > 1 else ""

            # split trainer/owner
            to = runner.get("trainer_owner", "").split("\n")
            trainer_name = to[0] if to else ""
            owner_name = to[1] if len(to) > 1 else ""

            records.append(
                {
                    # fixture-level
                    "fixture_index": fidx,
                    "fixture_day": fixture_day,
                    "fixture_type": fixture_type,
                    "racecourse": fixture.get("racecourse", ""),
                    "first_race_time": first_race_time,
                    "session": session,
                    "going_raw": going_raw,
                    "weather_raw": weather_raw,
                    "other_raw": other_raw,
                    # race-level
                    "race_index": ridx,
                    "race_time": race.get("time", ""),
                    "race_name": race.get("name", ""),
                    "race_distance_raw": race_distance,
                    "conditions": race.get("conditions", ""),
                    # runner-level
                    "runner_index": jidx,
                    "draw": runner.get("no_draw", "").split("\n")[0],
                    "horse_name": horse_name,
                    "jockey_name": jockey_name,
                    "age": runner.get("age", ""),
                    "form_type": runner.get("form_type", ""),
                    "bha_rating": runner.get("bha_rating", ""),
                    "weight_st_lb": runner.get("weight", ""),
                    "trainer_name": trainer_name,
                    "owner_name": owner_name,
                    "odds_raw": runner.get("odds", ""),
                }
            )

# build DataFrame
fixture_df = pd.DataFrame(records)

# apply conversions & parsing
fixture_df = fixture_df.assign(
    weather_category=lambda d: d["weather_raw"].apply(classify_weather),
    weight_kg=lambda d: d["weight_st_lb"].apply(wt_to_kg),
    starting_prob=lambda d: d["odds_raw"].apply(frac_to_prob),
    race_distance_m=lambda d: d["race_distance_raw"].apply(parse_distance_to_metres),
).drop(columns=["going_raw", "weather_raw", "other_raw", "odds_raw"])

# %%
