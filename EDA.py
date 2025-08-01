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


records = []
for file in sorted(RESULTS_DIR.glob("*.json")):
    ym = file.stem  # e.g. "2025_06"
    with file.open() as f:
        fixtures = json.load(f)
    for fixture in fixtures:
        # Pre-extract fixture-level features with defaults
        going_text = fixture.get("going", "")
        other_text = fixture.get("other_text", "")
        # Primary/secondary going
        glines = going_text.split("\n")
        primary_going = glines[1] if len(glines) > 1 else ""
        secondary_going = glines[2] if len(glines) > 2 else ""
        # Going stick
        m_stick = re.search(r"Going Stick\s*([\d\.]+)", going_text)
        going_stick = float(m_stick.group(1)) if m_stick else np.nan
        # Soil moisture
        m_soil = re.search(r"Soil Moisture:\s*(\d+)%", other_text)
        soil_moisture_pct = int(m_soil.group(1)) if m_soil else np.nan

        for race in fixture["races"]:
            for runner in race["runners"]:
                # Base record
                rec = {
                    "year_month": ym,
                    "fixture_index": fixture["fixture_index"],
                    "fixture_date": fixture["date"],
                    "fixture_year": fixture["year"],
                    "race_index": race["race_index"],
                    "racecourse": fixture["racecourse"],
                    # new going/soil features
                    "primary_going": primary_going,
                    "secondary_going": secondary_going,
                    "going_stick": going_stick,
                    "soil_moisture_pct": soil_moisture_pct,
                    # existing fields
                    "going": going_text,
                    "weather": fixture.get("weather", ""),
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

                # Parse position
                parts = rec["position_raw"].split("\n") if rec["position_raw"] else []
                rec["position_rank"] = re.sub(r"(?:st|nd|rd|th)$", "", parts[0]) if parts else ""
                rec["position_extra"] = parts[1] if len(parts) > 1 else ""
                m_draw = re.search(r"D:(\d+)", rec["position_extra"])
                rec["draw_number"] = int(m_draw.group(1)) if m_draw else np.nan

                # Horse/jockey & handicap
                hj_lines = rec["horse_jockey_raw"].split("\n")
                rec["horse_name"] = hj_lines[0] if hj_lines else ""
                rec["jockey_name"] = hj_lines[1] if len(hj_lines) > 1 else ""
                for line in hj_lines[2:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        rec[k.strip().lower().replace(" ", "_") + "_raw"] = v.strip()

                # Handicap fields
                raw_off = rec.pop("handicap_ran_off_raw", None)
                rec["handicap_ran_off"] = int(raw_off) if raw_off not in (None, "") else np.nan
                raw_perf = rec.pop("bha_performance_figure_raw", None)
                rec["bha_performance_figure"] = (
                    int(raw_perf) if raw_perf not in (None, "") else np.nan
                )
                raw_mark = rec.pop("current_handicap_mark_raw", "")
                mm = re.match(r"([A-Za-z]):\s*(\d+)", raw_mark)
                rec["current_mark_surface"] = mm.group(1) if mm else ""
                rec["current_mark"] = int(mm.group(2)) if mm else np.nan

                # Trainer/owner
                to_lines = rec["trainer_owner_raw"].split("\n")
                rec["trainer_name"] = to_lines[0] if to_lines else ""
                rec["owner_name"] = to_lines[1] if len(to_lines) > 1 else ""

                # Race distance
                rd = rec.pop("race_distance_raw", "")
                rd_val = rd.split("\n")[1].strip("()") if "\n" in rd else rd
                rec["race_distance"] = rd_val
                m_m = re.search(r"(\d+)m", rd_val)
                f_m = re.search(r"(\d+)f", rd_val)
                y_m = re.search(r"(\d+)y", rd_val)
                miles = int(m_m.group(1)) if m_m else 0
                furlongs = int(f_m.group(1)) if f_m else 0
                yards = int(y_m.group(1)) if y_m else 0
                rec["race_distance_m"] = miles * MILE_M + furlongs * FURLONG_M + yards * YARD_M

                # Distance/time
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
                tm = re.match(r"(\d+)m\s*([\d\.]+)s", raw_time)
                rec["finish_time_sec"] = (
                    (int(tm.group(1)) * 60 + float(tm.group(2))) if tm else np.nan
                )

                # SP
                rec["sp"] = rec.pop("sp_raw", "")

                records.append(rec)

# Build DataFrame
results_raw = pd.DataFrame(records)
race_raw = (
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
        "primary_going",
        "secondary_going",
    ]
)

raw_cols = [c for c in race_raw.columns if c.endswith("_raw")]

race_results = race_raw.drop(columns=raw_cols)

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


merged_df.set_index("race_date").to_csv(OUTPUT_DIR / "merged_results.csv")
# %%
# ? Fixture Loader


FIXTURES_JSON = OUTPUT_DIR / "fixtures.json"
with open(FIXTURES_JSON, "r", encoding="utf-8") as f:
    fixtures = json.load(f)

records = []
for fidx, fixture in enumerate(fixtures, start=1):
    going_txt = fixture.get("going", "")
    other_txt = fixture.get("other", "")
    weather_txt = fixture.get("weather", "")

    # primary/secondary going
    glines = going_txt.split("\n")
    primary_going = glines[1] if len(glines) > 1 else ""
    secondary_going = glines[2] if len(glines) > 2 else ""

    # going stick
    m_stick = re.search(r"Going Stick\s*([\d\.]+)", going_txt)
    going_stick = float(m_stick.group(1)) if m_stick else np.nan

    # soil moisture (in either going or other)
    combined = going_txt + "\n" + other_txt
    m_soil1 = re.search(r"([\d\.]+)%\s*soil moisture", combined, flags=re.IGNORECASE)
    m_soil2 = re.search(r"Soil Moisture[:\s]*([\d\.]+)%", combined, flags=re.IGNORECASE)
    soil_moisture_pct = (
        float(m_soil1.group(1)) if m_soil1 else float(m_soil2.group(1)) if m_soil2 else np.nan
    )

    weather_cat = classify_weather(weather_txt)
    day_str = fixture.get("type_date", "").split("\n", 1)[0]  # e.g. "SAT 02 AUG"

    for ridx, race in enumerate(fixture.get("races", []), start=1):
        rd_raw = race.get("distance", "")
        race_dist_str = rd_raw.split("\n", 1)[-1].strip("()")

        for jidx, runner in enumerate(race.get("runners", []), start=1):
            hj = runner.get("horse_jockey", "").split("\n")
            to = runner.get("trainer_owner", "").split("\n")
            form = runner.get("form_type", "")

            # current_mark_surface and current_mark from the same regex
            m_mark = re.search(r"([A-Za-z]):\s*(\d+)", form)
            if m_mark:
                current_mark_surface = m_mark.group(1)
                current_mark = int(m_mark.group(2))
            else:
                current_mark_surface = ""
                current_mark = np.nan

            # handicap ran off
            m_h = re.search(r"H:\s*(\d+)", form)
            handicap_ran_off = int(m_h.group(1)) if m_h else np.nan

            # bha_performance_figure
            bha_perf = pd.to_numeric(runner.get("bha_rating", ""), errors="coerce")

            # draw_number
            m_draw = re.search(r"(\d+)", runner.get("no_draw", ""))
            draw_number = int(m_draw.group(1)) if m_draw else 0

            records.append(
                {
                    "fixture_index": fidx,
                    "race_index": ridx,
                    "racecourse": fixture.get("racecourse", ""),
                    "primary_going": primary_going,
                    "secondary_going": secondary_going,
                    "going_stick": going_stick,
                    "soil_moisture_pct": soil_moisture_pct,
                    "runner_index": jidx,
                    "draw_number": draw_number,
                    "horse_name": hj[0] if hj else "",
                    "jockey_name": hj[1] if len(hj) > 1 else "",
                    "handicap_ran_off": handicap_ran_off,
                    "bha_performance_figure": bha_perf,
                    "current_mark_surface": current_mark_surface,
                    "current_mark": current_mark,
                    "trainer_name": to[0] if to else "",
                    "owner_name": to[1] if len(to) > 1 else "",
                    "race_distance_m": parse_distance_to_metres(race_dist_str),
                    "race_date": pd.to_datetime(
                        f"{day_str} 2025 {race.get('time', '')}",
                        format="%a %d %b %Y %I:%M%p",
                        dayfirst=True,
                    ),
                    "weather_category": weather_cat,
                    "DNF": False,
                    "WD": False,
                    "NR": False,
                }
            )

# assemble DataFrame & order columns to match race_results

fixture_results = pd.DataFrame(records).drop(columns=["primary_going", "secondary_going"])

fixture_output = (
    fixture_results.merge(horse_df, on="horse_name", how="left")
    .merge(jockey_df, on="jockey_name", how="left")
    .assign(race_age=lambda df: (df["race_date"] - df["foaled"]).dt.days)
)

fixture_output.set_index("race_date").to_csv(OUTPUT_DIR / "fixture_results.csv")


# %%

# %%
