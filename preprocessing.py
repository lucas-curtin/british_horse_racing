"""Build race results, leakage-safe historical features, and snapshot outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re

from model_feature_spec import (
    MODEL_INPUTS_DIR,
    NON_RUNNER_EVENTS_CSV,
    SEQUENTIAL_MODEL_INPUT_CSV,
    allowed_model_input_columns,
    validate_model_input_frame,
)

pd.options.future.no_silent_downcasting = True

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
RACE_RESULTS_CSV = OUTPUT_DIR / "race_results.csv"
HISTORICAL_FEATURES_CSV = OUTPUT_DIR / "historical_features.csv"
HORSE_HISTORY_CSV = OUTPUT_DIR / "horse_df.csv"
JOCKEY_HISTORY_CSV = OUTPUT_DIR / "jockey_df.csv"

MILE_M = 1609.34
FURLONG_M = 201.168
YARD_M = 0.9144
RATINGS_CATEGORIES = ("flat", "chase", "hurdle", "awt")
DISTANCE_BUCKET_LABELS = ("short", "medium", "staying", "marathon")
RACE_KEYWORDS = {
    "handicap": ["handicap"],
    "steeple": ["steeple"],
    "chase": ["chase"],
    "novice": ["novice", "novices'"],
    "hurdle": ["hurdle"],
    "maiden": ["maiden"],
    "national_hunt": ["nh flat", "national hunt", "bumper"],
    "selling": ["selling", "seller"],
}
NON_RUNNER_REASON_PATTERNS = {
    "injury_or_self_cert": [
        r"self cert",
        r"injur",
        r"bruise",
        r"heat in",
        r"leg",
        r"lame",
        r"sore",
    ],
    "going": [
        r"\bgoing\b",
        r"\bground\b",
    ],
    "vet_or_medical": [
        r"\bvet\b",
        r"\bmedical\b",
        r"\bill\b",
        r"\btemperature\b",
    ],
    "administrative": [
        r"steward",
        r"transport",
        r"passport",
        r"travel",
        r"administrative",
    ],
}


def classify_weather(description: str) -> str:
    """Classify a free-text weather description into a simple category."""
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
    """Convert a `st lbs` weight string to kilograms."""
    parts = weight.replace("lbs", "").split("st")
    if len(parts) != 2:
        return np.nan
    try:
        stones = int(parts[0])
        pounds = int(parts[1].strip())
    except ValueError:
        return np.nan
    return stones * 6.35029 + pounds * 0.453592


def frac_to_prob(odds: str) -> float:
    """Convert fractional odds into an implied probability."""
    if not isinstance(odds, str) or not odds.strip():
        return np.nan

    parts = odds.split("/")
    if len(parts) != 2:
        return np.nan

    num_str, denom_str = parts
    if not num_str.replace(".", "", 1).isdigit() or not denom_str.replace(".", "", 1).isdigit():
        return np.nan

    numerator = float(num_str)
    denominator = float(denom_str)
    if numerator < 0 or denominator <= 0:
        return np.nan
    return denominator / (numerator + denominator)


def parse_currency(value: str) -> float:
    """Parse a GBP currency string into a float."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    cleaned = value.replace("£", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_race_distance(distance_text: str) -> float:
    """Convert a race distance string into metres."""
    miles_match = re.search(r"(\d+)m", distance_text)
    furlongs_match = re.search(r"(\d+)f", distance_text)
    yards_match = re.search(r"(\d+)y", distance_text)
    miles = int(miles_match.group(1)) if miles_match else 0
    furlongs = int(furlongs_match.group(1)) if furlongs_match else 0
    yards = int(yards_match.group(1)) if yards_match else 0
    return miles * MILE_M + furlongs * FURLONG_M + yards * YARD_M


def parse_position_rank(position_raw: str) -> tuple[str, float]:
    """Return the textual rank and numeric finish position when available."""
    if not position_raw:
        return "", np.nan
    rank = position_raw.split("\n", maxsplit=1)[0].strip()
    numeric = re.sub(r"(?:st|nd|rd|th)$", "", rank)
    return rank, float(numeric) if numeric.isdigit() else np.nan


def parse_draw_number(position_raw: str) -> float:
    """Extract drawn stall info like `D:10` from the raw position cell."""
    draw_match = re.search(r"\bD:(\d+)\b", position_raw or "")
    return float(draw_match.group(1)) if draw_match else np.nan


def parse_current_mark(raw_mark: str) -> tuple[str, float]:
    """Parse current handicap mark text like `H:121`."""
    match = re.match(r"([A-Za-z]):\s*(\d+)", raw_mark or "")
    if not match:
        return "", np.nan
    return match.group(1), float(match.group(2))


def parse_fixture_date(date_text: str, year: int) -> pd.Timestamp:
    """Parse fixture date text into a timestamp."""
    return pd.to_datetime(f"{date_text} {year}", dayfirst=True, errors="coerce")


def parse_declared_at(declared_at: str, year: int) -> pd.Timestamp:
    """Parse non-runner declaration text like `Thu 26 Mar 1:37pm`."""
    if not declared_at:
        return pd.NaT
    return pd.to_datetime(
        f"{declared_at} {year}",
        format="%a %d %b %I:%M%p %Y",
        errors="coerce",
    )


def classify_non_runner_reason(reason: str) -> str:
    """Map free-text non-runner reasons into a small category set."""
    lowered_reason = reason.lower().strip()
    if any(re.search(pattern, lowered_reason) for pattern in NON_RUNNER_REASON_PATTERNS["going"]):
        return "going"
    for category, patterns in NON_RUNNER_REASON_PATTERNS.items():
        if category == "going":
            continue
        if any(re.search(pattern, lowered_reason) for pattern in patterns):
            return category
    return "other"


def build_fixture_context(fixture: dict[str, object]) -> dict[str, object]:
    """Extract reusable fixture-level context."""
    going_text = str(fixture.get("going", ""))
    going_lines = going_text.split("\n")
    primary_going = going_lines[1] if len(going_lines) > 1 else ""
    secondary_going = going_lines[2] if len(going_lines) > 2 else ""

    going_stick_match = re.search(r"Going Stick\s*([\d\.]+)", going_text)
    going_stick = float(going_stick_match.group(1)) if going_stick_match else np.nan

    soil_moisture_pct = np.nan
    for line in going_text.splitlines():
        if "moisture" not in line.lower():
            continue
        soil_match = re.search(r"([\d.]+)%", line)
        if soil_match:
            soil_moisture_pct = float(soil_match.group(1))
            break

    fixture_timestamp = parse_fixture_date(
        str(fixture.get("date", "")),
        int(fixture.get("year", 0)),
    )
    return {
        "fixture_timestamp": fixture_timestamp,
        "going_stick": going_stick,
        "soil_moisture_pct": soil_moisture_pct,
        "primary_going": primary_going,
        "secondary_going": secondary_going,
        "weather_category": classify_weather(str(fixture.get("weather", ""))),
        "fixture_index": fixture.get("fixture_index", ""),
        "racecourse": fixture.get("racecourse", ""),
    }


def build_race_context(
    fixture_context: dict[str, object],
    race: dict[str, object],
) -> dict[str, object]:
    """Extract reusable race-level context."""
    fixture_timestamp = fixture_context["fixture_timestamp"]
    race_details = race.get("race_details", {})
    race_distance_raw = str(race.get("distance", ""))
    race_distance = str(race_details.get("Race Distance", race_distance_raw)).replace("\n", " ")
    race_distance_m = parse_race_distance(race_distance)
    race_time_raw = str(race.get("time", ""))
    race_datetime = pd.to_datetime(
        fixture_timestamp.strftime("%Y-%m-%d") + f" {race_time_raw}",
        format="%Y-%m-%d %I:%M%p",
        errors="coerce",
    )
    if pd.isna(race_datetime):
        race_datetime = fixture_timestamp + pd.to_timedelta(
            int(race.get("race_index", 0)),
            unit="m",
        )

    distance_bucket = pd.cut(
        pd.Series([race_distance_m]),
        bins=[0, 2400, 3600, 5200, np.inf],
        labels=DISTANCE_BUCKET_LABELS,
        include_lowest=True,
    ).astype(str).iloc[0]
    return {
        "race_id": (
            fixture_timestamp.strftime("%Y%m%d")
            + "_"
            + str(fixture_context["fixture_index"])
            + "_"
            + str(race.get("race_index", ""))
        ),
        "race_index": race.get("race_index", ""),
        "race_date": fixture_timestamp,
        "race_datetime": race_datetime,
        "race_time_raw": race_time_raw,
        "race_distance_raw": race_distance_raw,
        "race_distance_m": race_distance_m,
        "distance_bucket": distance_bucket,
        "race_name_raw": race.get("name", ""),
        "winner_info_raw": race.get("winner_info", ""),
        "race_going": str(race_details.get("Race Going", fixture_context["primary_going"] or "")),
        "race_runners": race_details.get("Runners", ""),
        "race_prize_money": parse_currency(str(race_details.get("Prize Money", ""))),
        "race_age_band": race_details.get("Horse Age", ""),
        "race_type_detail": race_details.get("Horse Type", ""),
        "race_handicapper": race_details.get("Handicapper", ""),
        "race_rating_band": race_details.get("Rating", ""),
        "race_min_weight": race_details.get("Min Weight", ""),
        "race_weights_raised": race_details.get("Weights Raised", ""),
        "race_rider_type": race_details.get("Rider Type", ""),
    }


def build_non_runner_events() -> pd.DataFrame:
    """Flatten fixture-level non-runner entries into an audit table."""
    records: list[dict[str, object]] = []
    for file in sorted(RESULTS_DIR.glob("*.json")):
        with file.open(encoding="utf-8") as handle:
            fixtures = json.load(handle)

        for fixture in fixtures:
            fixture_context = build_fixture_context(fixture)
            fixture_timestamp = fixture_context["fixture_timestamp"]
            fixture_year = int(fixture.get("year", 0))

            for entry in fixture.get("non_runners", []):
                race_time_raw = str(entry.get("race_time", "")).strip()
                event_datetime = pd.to_datetime(
                    fixture_timestamp.strftime("%Y-%m-%d") + f" {race_time_raw.upper()}",
                    format="%Y-%m-%d %I:%M%p",
                    errors="coerce",
                )
                if pd.isna(event_datetime):
                    event_datetime = fixture_timestamp

                reason_raw = str(entry.get("reason", "")).strip()
                declared_at_raw = str(entry.get("declared_at", "")).strip()
                records.append(
                    {
                        "fixture_index": fixture_context["fixture_index"],
                        "race_date": fixture_timestamp,
                        "racecourse": fixture_context["racecourse"],
                        "race_time_raw": race_time_raw,
                        "event_datetime": event_datetime,
                        "horse_name": str(entry.get("horse_name", "")).strip(),
                        "declared_at_raw": declared_at_raw,
                        "declared_at": parse_declared_at(declared_at_raw, fixture_year),
                        "reason_raw": reason_raw,
                        "reason_category": classify_non_runner_reason(reason_raw),
                    },
                )

    non_runner_events = pd.DataFrame(records)
    if non_runner_events.empty:
        return non_runner_events

    return non_runner_events.sort_values(
        ["horse_name", "event_datetime", "racecourse", "race_time_raw"],
        kind="stable",
    ).reset_index(drop=True)


def parse_runner_identity(
    runner: dict[str, object],
) -> dict[str, object]:
    """Parse runner-level identity fields from raw and structured table data."""
    horse_jockey_raw = str(runner.get("horse_jockey", ""))
    trainer_owner_raw = str(runner.get("trainer_owner", ""))
    table_data = runner.get("table_data", {})
    horse_jockey_lines = [line for line in horse_jockey_raw.split("\n") if line]
    trainer_owner_lines = [line for line in trainer_owner_raw.split("\n") if line]

    horse_name = str(table_data.get("horse_name", "")) or (
        horse_jockey_lines[0] if horse_jockey_lines else ""
    )
    jockey_name = str(table_data.get("jockey_name", "")) or (
        horse_jockey_lines[1] if len(horse_jockey_lines) > 1 else ""
    )
    trainer_name = str(table_data.get("trainer_name", "")) or (
        trainer_owner_lines[0] if trainer_owner_lines else ""
    )
    owner_name = str(table_data.get("owner_name", "")) or (
        trainer_owner_lines[1] if len(trainer_owner_lines) > 1 else ""
    )

    raw_mark = ""
    for line in horse_jockey_lines[2:]:
        if line.lower().startswith("current handicap mark:"):
            raw_mark = line.split(":", maxsplit=1)[1].strip()
            break
    if not raw_mark:
        mark_text = str(table_data.get("handicap_mark_text", ""))
        raw_mark = mark_text.replace("Current handicap mark:", "").strip()

    current_mark_surface, current_mark = parse_current_mark(raw_mark)
    return {
        "horse_name": horse_name,
        "horse_url": table_data.get("horse_url", "nan"),
        "jockey_name": jockey_name,
        "jockey_url": table_data.get("jockey_url", "nan"),
        "trainer_name": trainer_name,
        "trainer_url": table_data.get("trainer_url", "nan"),
        "owner_name": owner_name,
        "horse_jockey_raw": horse_jockey_raw,
        "trainer_owner_raw": trainer_owner_raw,
        "current_mark_surface": current_mark_surface,
        "current_mark": current_mark,
    }


def parse_runner_timing(runner: dict[str, object]) -> dict[str, object]:
    """Parse runner distance and timing fields."""
    table_data = runner.get("table_data", {})
    distance_time_raw = str(runner.get("distance_time", ""))
    distance_lines = [line for line in distance_time_raw.split("\n") if line]

    if len(distance_lines) >= 2:
        distance_raw = distance_lines[0]
        finish_time_raw = distance_lines[-1]
    elif len(distance_lines) == 1 and re.search(r"\d+m\s*[\d\.]+s", distance_lines[0]):
        distance_raw = ""
        finish_time_raw = distance_lines[0]
    elif len(distance_lines) == 1:
        distance_raw = distance_lines[0]
        finish_time_raw = ""
    else:
        distance_raw = str(table_data.get("distance_text", ""))
        finish_time_raw = str(table_data.get("finish_time", ""))

    timing_match = re.match(r"(\d+)m\s*([\d\.]+)s", finish_time_raw)
    finish_time_sec = (
        int(timing_match.group(1)) * 60 + float(timing_match.group(2))
        if timing_match
        else np.nan
    )
    return {
        "distance_time_raw": distance_time_raw,
        "distance_raw": distance_raw,
        "finish_time_raw": finish_time_raw,
        "finish_time_sec": finish_time_sec,
    }


def build_runner_record(
    fixture_context: dict[str, object],
    race_context: dict[str, object],
    runner: dict[str, object],
) -> dict[str, object]:
    """Build one flattened runner record."""
    position_raw = str(runner.get("position", ""))
    position_rank, finish_position = parse_position_rank(position_raw)
    identity = parse_runner_identity(runner)
    timing = parse_runner_timing(runner)
    return {
        **fixture_context,
        **race_context,
        **identity,
        **timing,
        "runner_index": runner.get("runner_index", ""),
        "position_rank": position_rank,
        "finish_position": finish_position,
        "draw_number": parse_draw_number(position_raw),
        "starting_prob": frac_to_prob(str(runner.get("sp", ""))),
        "won": bool(pd.notna(finish_position) and finish_position == 1),
        "placed": bool(pd.notna(finish_position) and finish_position <= 3),
        "DNF": position_rank == "DNF",
        "WD": position_rank == "WD",
        "NR": position_rank == "NR",
        "position_raw": position_raw,
    }


def build_race_results() -> pd.DataFrame:
    """Flatten fixture results JSON into a race-results table."""
    records: list[dict[str, object]] = []
    for file in sorted(RESULTS_DIR.glob("*.json")):
        with file.open(encoding="utf-8") as handle:
            fixtures = json.load(handle)

        for fixture in fixtures:
            fixture_context = build_fixture_context(fixture)
            for race in fixture.get("races", []):
                race_context = build_race_context(fixture_context, race)
                records.extend(
                    build_runner_record(fixture_context, race_context, runner)
                    for runner in race.get("runners", [])
                )

    race_results = pd.DataFrame(records)
    keyword_flags = (
        race_results["race_name_raw"]
        .apply(lambda name: classify_race_keywords(str(name), RACE_KEYWORDS))
        .apply(pd.Series)
    )
    race_results = pd.concat([race_results, keyword_flags], axis=1)
    race_results["class"] = pd.to_numeric(race_results["class"], errors="coerce")
    race_results["speed"] = race_results["race_distance_m"] / race_results["finish_time_sec"]
    race_results["log_speed"] = np.log(race_results["speed"])
    return race_results.sort_values(
        ["race_datetime", "race_id", "runner_index"],
        kind="stable",
    ).reset_index(drop=True)


def classify_race_keywords(
    name: str,
    keywords: dict[str, list[str]],
) -> dict[str, bool | int | None]:
    """Return binary race flags plus extracted class number."""
    lowered_name = name.lower()
    flags = {
        flag: any(keyword in lowered_name for keyword in values)
        for flag, values in keywords.items()
    }
    match = re.search(r"class\s*(\d)", lowered_name)
    flags["class"] = int(match.group(1)) if match else None
    return flags


def prior_cumsum(series: pd.Series) -> pd.Series:
    """Return a shifted cumulative sum."""
    return series.fillna(0).astype(float).shift(fill_value=0).cumsum()


def prior_expanding_mean(series: pd.Series) -> pd.Series:
    """Return the expanding mean using only prior observations."""
    return series.shift().expanding().mean()


def add_context_group_features(
    df: pd.DataFrame,
    entity_col: str,
    context_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Add prior run and win counts for an entity in a specific context."""
    grouped = df.groupby([entity_col, context_col], sort=False)
    df[f"{prefix}_runs"] = grouped["is_valid_run"].transform(prior_cumsum)
    df[f"{prefix}_wins"] = grouped["won"].transform(prior_cumsum)
    return df


def prior_known_value(series: pd.Series) -> pd.Series:
    """Return the most recent non-empty known value before the current row."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        cleaned = series.mask(series == "", np.nan)
    else:
        cleaned = series
    return cleaned.ffill().infer_objects(copy=False).shift()


def add_actor_features(
    df: pd.DataFrame,
    entity_col: str,
    prefix: str,
    runs_label: str,
    *,
    include_days_since_win: bool = False,
) -> pd.DataFrame:
    """Add leakage-safe prior performance features for one actor."""
    grouped = df.groupby(entity_col, sort=False)
    prior_runs = grouped["is_valid_run"].transform(prior_cumsum)
    prior_wins = grouped["won"].transform(prior_cumsum)
    prior_places = grouped["placed"].transform(prior_cumsum)
    df[f"{prefix}_{runs_label}"] = prior_runs
    df[f"{prefix}_wins"] = prior_wins
    df[f"{prefix}_places"] = prior_places
    df[f"{prefix}_win_rate"] = np.where(prior_runs > 0, prior_wins / prior_runs, np.nan)
    df[f"{prefix}_place_rate"] = np.where(prior_runs > 0, prior_places / prior_runs, np.nan)
    df[f"{prefix}_avg_finish_position"] = grouped["finish_position"].transform(prior_expanding_mean)
    df[f"{prefix}_avg_speed"] = grouped["speed"].transform(prior_expanding_mean)
    df[f"{prefix}_avg_finish_time_sec"] = grouped["finish_time_sec"].transform(prior_expanding_mean)

    prior_valid_datetime = grouped["race_datetime"].transform(
        lambda series: series.where(df.loc[series.index, "is_valid_run"]).ffill().shift(),
    )
    df[f"{prefix}_days_since_last_{runs_label[:-1]}"] = (
        df["race_datetime"] - prior_valid_datetime
    ).dt.days

    if include_days_since_win:
        prior_win_datetime = grouped["race_datetime"].transform(
            lambda series: series.where(df.loc[series.index, "won"]).ffill().shift(),
        )
        df[f"{prefix}_days_since_last_win"] = (df["race_datetime"] - prior_win_datetime).dt.days

    return df


def build_historical_features(race_results: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe historical features from prior race results only."""
    features = race_results.copy()
    features["is_valid_run"] = ~(features["NR"] | features["WD"])
    features["horse_jockey_pair"] = (
        features["horse_name"].fillna("")
        + "__"
        + features["jockey_name"].fillna("")
    )

    features = add_actor_features(features, "horse_name", "horse_prior", "runs")
    features = add_actor_features(
        features,
        "jockey_name",
        "jockey_prior",
        "rides",
        include_days_since_win=True,
    )
    features = add_actor_features(features, "trainer_name", "trainer_prior", "runs")
    features = add_actor_features(features, "owner_name", "owner_prior", "runs")
    features = add_actor_features(features, "horse_jockey_pair", "pair_prior", "rides")

    features = add_context_group_features(
        features,
        "horse_name",
        "racecourse",
        "horse_prior_course",
    )
    features = add_context_group_features(
        features,
        "horse_name",
        "race_going",
        "horse_prior_going",
    )
    features = add_context_group_features(
        features,
        "horse_name",
        "distance_bucket",
        "horse_prior_distance_bucket",
    )
    features = add_context_group_features(
        features,
        "jockey_name",
        "racecourse",
        "jockey_prior_course",
    )
    features = add_context_group_features(
        features,
        "jockey_name",
        "race_going",
        "jockey_prior_going",
    )
    features = add_context_group_features(
        features,
        "jockey_name",
        "distance_bucket",
        "jockey_prior_distance_bucket",
    )

    features["horse_prior_mark_change"] = features.groupby("horse_name", sort=False)[
        "current_mark"
    ].transform(lambda series: series.shift() - series.shift(2))
    features["horse_prior_current_mark"] = features.groupby("horse_name", sort=False)[
        "current_mark"
    ].transform(prior_known_value)
    features["horse_prior_current_mark_surface"] = features.groupby("horse_name", sort=False)[
        "current_mark_surface"
    ].transform(prior_known_value)
    features["horse_prior_trainer_name"] = features.groupby("horse_name", sort=False)[
        "trainer_name"
    ].transform(prior_known_value)
    features["horse_prior_owner_name"] = features.groupby("horse_name", sort=False)[
        "owner_name"
    ].transform(prior_known_value)

    return features.drop(columns=["is_valid_run"]).reset_index(drop=True)


def merge_prior_event_state(
    race_results: pd.DataFrame,
    event_frame: pd.DataFrame,
    *,
    event_time_col: str,
    event_value_cols: list[str],
) -> pd.DataFrame:
    """Attach the latest prior horse event state to each race row."""
    if event_frame.empty:
        return race_results.copy()

    merged_groups: list[pd.DataFrame] = []
    for horse_name, horse_race_rows in race_results.groupby("horse_name", sort=False):
        horse_events = event_frame.loc[event_frame["horse_name"].eq(horse_name)].copy()
        horse_races = horse_race_rows.sort_values("race_datetime", kind="stable").copy()
        if horse_events.empty:
            for column in event_value_cols:
                horse_races[column] = np.nan
            merged_groups.append(horse_races)
            continue

        horse_events = horse_events.sort_values(event_time_col, kind="stable")
        merge_columns = list(dict.fromkeys(["horse_name", event_time_col, *event_value_cols]))
        merged = pd.merge_asof(
            horse_races,
            horse_events.loc[:, merge_columns],
            left_on="race_datetime",
            right_on=event_time_col,
            by="horse_name",
            direction="backward",
            allow_exact_matches=False,
        )
        if event_time_col not in event_value_cols:
            merged = merged.drop(columns=[event_time_col])
        merged_groups.append(merged)

    return (
        pd.concat(merged_groups, axis=0)
        .sort_values(["race_datetime", "race_id", "runner_index"], kind="stable")
        .reset_index(drop=True)
    )


def add_non_runner_history_features(
    race_results: pd.DataFrame,
    non_runner_events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach prior non-runner history features to race rows."""
    features = race_results.copy()
    if non_runner_events.empty:
        zero_fill_columns = [
            "horse_prior_non_runner_count",
            "horse_prior_injury_non_runner_count",
            "horse_prior_going_non_runner_count",
            "horse_days_since_last_non_runner",
            "horse_days_since_last_injury_non_runner",
        ]
        for column in zero_fill_columns:
            features[column] = np.nan if "days_since" in column else 0.0
        return features

    event_state = non_runner_events.copy()
    event_state["is_injury_non_runner"] = event_state["reason_category"].eq(
        "injury_or_self_cert",
    ).astype(int)
    event_state["is_going_non_runner"] = event_state["reason_category"].eq("going").astype(int)
    grouped = event_state.groupby("horse_name", sort=False)
    event_state["horse_prior_non_runner_count"] = grouped.cumcount() + 1
    event_state["horse_prior_injury_non_runner_count"] = grouped[
        "is_injury_non_runner"
    ].cumsum()
    event_state["horse_prior_going_non_runner_count"] = grouped["is_going_non_runner"].cumsum()
    event_state["last_non_runner_datetime"] = event_state["event_datetime"]

    features = merge_prior_event_state(
        features,
        event_state,
        event_time_col="event_datetime",
        event_value_cols=[
            "horse_prior_non_runner_count",
            "horse_prior_injury_non_runner_count",
            "horse_prior_going_non_runner_count",
            "last_non_runner_datetime",
        ],
    )

    injury_events = event_state.loc[
        event_state["is_injury_non_runner"].eq(1),
        ["horse_name", "event_datetime"],
    ].rename(columns={"event_datetime": "last_injury_non_runner_datetime"})
    features = merge_prior_event_state(
        features,
        injury_events,
        event_time_col="last_injury_non_runner_datetime",
        event_value_cols=["last_injury_non_runner_datetime"],
    )

    features["horse_prior_non_runner_count"] = (
        features["horse_prior_non_runner_count"].fillna(0).astype(float)
    )
    features["horse_prior_injury_non_runner_count"] = (
        features["horse_prior_injury_non_runner_count"].fillna(0).astype(float)
    )
    features["horse_prior_going_non_runner_count"] = (
        features["horse_prior_going_non_runner_count"].fillna(0).astype(float)
    )
    features["last_non_runner_datetime"] = pd.to_datetime(
        features["last_non_runner_datetime"],
        errors="coerce",
    )
    features["last_injury_non_runner_datetime"] = pd.to_datetime(
        features["last_injury_non_runner_datetime"],
        errors="coerce",
    )
    features["horse_days_since_last_non_runner"] = (
        features["race_datetime"] - features["last_non_runner_datetime"]
    ).dt.days
    features["horse_days_since_last_injury_non_runner"] = (
        features["race_datetime"] - features["last_injury_non_runner_datetime"]
    ).dt.days
    return features.drop(
        columns=["last_non_runner_datetime", "last_injury_non_runner_datetime"],
    )


def add_dnf_history_features(race_results: pd.DataFrame) -> pd.DataFrame:
    """Attach prior DNF history features to race rows."""
    features = race_results.copy()
    grouped = features.groupby("horse_name", sort=False)
    features["horse_prior_dnf_count"] = grouped["DNF"].transform(prior_cumsum)
    prior_dnf_datetime = grouped["race_datetime"].transform(
        lambda series: series.where(features.loc[series.index, "DNF"]).ffill().shift(),
    )
    features["horse_days_since_last_dnf"] = (
        features["race_datetime"] - prior_dnf_datetime
    ).dt.days
    return features


def build_sequential_model_input(
    race_results: pd.DataFrame,
    non_runner_events: pd.DataFrame,
) -> pd.DataFrame:
    """Build the curated, allow-listed dataset for sequential ranking models."""
    model_input = add_dnf_history_features(race_results)
    model_input = add_non_runner_history_features(model_input, non_runner_events)
    model_input["horse_course_key"] = (
        model_input["horse_name"].fillna("")
        + "__"
        + model_input["racecourse"].fillna("")
    )
    model_input["horse_going_key"] = (
        model_input["horse_name"].fillna("")
        + "__"
        + model_input["race_going"].fillna("")
    )
    model_input["horse_jockey_key"] = (
        model_input["horse_name"].fillna("")
        + "__"
        + model_input["jockey_name"].fillna("")
    )
    model_input = model_input.loc[(~model_input["NR"]) & (~model_input["WD"])].copy()
    model_input = model_input.sort_values(
        ["race_datetime", "race_id", "runner_index"],
        kind="stable",
    ).reset_index(drop=True)
    model_input = model_input.loc[:, allowed_model_input_columns()]
    validate_model_input_frame(model_input)
    return model_input


def build_entity_history(
    entity_results: pd.DataFrame,
    entity_col: str,
    columns: list[str],
) -> pd.DataFrame:
    """Build an end-of-day entity-date history from the last race row on each date."""
    history = (
        entity_results.loc[:, [entity_col, "race_date", "race_datetime", "race_id", *columns]]
        .dropna(subset=[entity_col])
        .sort_values([entity_col, "race_datetime", "race_id"], kind="stable")
        .groupby([entity_col, "race_date"], sort=False, as_index=False)
        .last()
        .rename(columns={"race_date": "record_date"})
    )
    return history.drop(columns=["race_datetime", "race_id"])


def build_horse_history_df(race_results: pd.DataFrame) -> pd.DataFrame:
    """Build an end-of-day horse-date history table from race results."""
    horse_results = race_results.copy()
    horse_grouped = horse_results.groupby("horse_name", sort=False)
    cumulative_runs = horse_grouped["horse_name"].cumcount() + 1
    cumulative_wins = horse_grouped["won"].transform(
        lambda series: series.astype(int).cumsum(),
    )
    cumulative_places = horse_grouped["placed"].transform(
        lambda series: series.astype(int).cumsum(),
    )

    horse_results["horse_runs"] = cumulative_runs.astype(float)
    horse_results["horse_wins"] = cumulative_wins.astype(float)
    horse_results["horse_places"] = cumulative_places.astype(float)
    horse_results["horse_win_rate"] = np.where(
        cumulative_runs > 0,
        cumulative_wins / cumulative_runs,
        np.nan,
    )
    horse_results["horse_place_rate"] = np.where(
        cumulative_runs > 0,
        cumulative_places / cumulative_runs,
        np.nan,
    )
    horse_results["horse_avg_finish_position"] = horse_grouped["finish_position"].transform(
        lambda series: series.expanding().mean(),
    )
    horse_results["horse_avg_speed"] = horse_grouped["speed"].transform(
        lambda series: series.expanding().mean(),
    )
    horse_results["horse_avg_finish_time_sec"] = horse_grouped["finish_time_sec"].transform(
        lambda series: series.expanding().mean(),
    )
    horse_results["horse_days_since_last_run"] = (
        horse_results["race_datetime"] - horse_grouped["race_datetime"].shift()
    ).dt.days
    horse_results["horse_mark_change"] = horse_grouped["current_mark"].transform(
        lambda series: series - series.shift(),
    )

    columns = [
        "horse_url",
        "horse_runs",
        "horse_wins",
        "horse_places",
        "horse_win_rate",
        "horse_place_rate",
        "horse_avg_finish_position",
        "horse_avg_speed",
        "horse_avg_finish_time_sec",
        "horse_days_since_last_run",
        "horse_mark_change",
        "current_mark",
        "current_mark_surface",
        "trainer_name",
        "owner_name",
        "position_rank",
        "finish_position",
        "DNF",
        "NR",
        "WD",
        "finish_time_sec",
        "starting_prob",
        "racecourse",
        "race_going",
        "distance_bucket",
    ]
    history = build_entity_history(horse_results, "horse_name", columns).rename(
        columns={
            "current_mark": "horse_current_mark",
            "current_mark_surface": "horse_current_mark_surface",
            "trainer_name": "horse_current_trainer_name",
            "owner_name": "horse_current_owner_name",
            "position_rank": "horse_last_position_rank",
            "finish_position": "horse_last_finish_position",
            "DNF": "horse_last_DNF",
            "NR": "horse_last_NR",
            "WD": "horse_last_WD",
            "finish_time_sec": "horse_last_finish_time_sec",
            "starting_prob": "horse_last_starting_prob",
            "racecourse": "horse_last_racecourse",
            "race_going": "horse_last_race_going",
            "distance_bucket": "horse_last_distance_bucket",
        },
    )
    return history.sort_values(["horse_name", "record_date"], kind="stable").reset_index(drop=True)


def build_jockey_history_df(race_results: pd.DataFrame) -> pd.DataFrame:
    """Build an end-of-day jockey-date history table from race results."""
    jockey_results = race_results.copy()
    jockey_grouped = jockey_results.groupby("jockey_name", sort=False)
    cumulative_rides = jockey_grouped["jockey_name"].cumcount() + 1
    cumulative_wins = jockey_grouped["won"].transform(
        lambda series: series.astype(int).cumsum(),
    )
    cumulative_places = jockey_grouped["placed"].transform(
        lambda series: series.astype(int).cumsum(),
    )

    jockey_results["jockey_rides"] = cumulative_rides.astype(float)
    jockey_results["jockey_wins"] = cumulative_wins.astype(float)
    jockey_results["jockey_places"] = cumulative_places.astype(float)
    jockey_results["jockey_win_rate"] = np.where(
        cumulative_rides > 0,
        cumulative_wins / cumulative_rides,
        np.nan,
    )
    jockey_results["jockey_place_rate"] = np.where(
        cumulative_rides > 0,
        cumulative_places / cumulative_rides,
        np.nan,
    )
    jockey_results["jockey_avg_finish_position"] = jockey_grouped["finish_position"].transform(
        lambda series: series.expanding().mean(),
    )
    jockey_results["jockey_avg_speed"] = jockey_grouped["speed"].transform(
        lambda series: series.expanding().mean(),
    )
    jockey_results["jockey_avg_finish_time_sec"] = jockey_grouped["finish_time_sec"].transform(
        lambda series: series.expanding().mean(),
    )
    jockey_results["jockey_days_since_last_ride"] = (
        jockey_results["race_datetime"] - jockey_grouped["race_datetime"].shift()
    ).dt.days
    prior_win_datetime = jockey_grouped["race_datetime"].transform(
        lambda series: series.where(jockey_results.loc[series.index, "won"]).ffill().shift(),
    )
    jockey_results["jockey_days_since_last_win"] = (
        jockey_results["race_datetime"] - prior_win_datetime
    ).dt.days

    columns = [
        "jockey_url",
        "jockey_rides",
        "jockey_wins",
        "jockey_places",
        "jockey_win_rate",
        "jockey_place_rate",
        "jockey_avg_finish_position",
        "jockey_avg_speed",
        "jockey_avg_finish_time_sec",
        "jockey_days_since_last_ride",
        "jockey_days_since_last_win",
        "position_rank",
        "finish_position",
        "DNF",
        "NR",
        "WD",
        "finish_time_sec",
        "starting_prob",
        "racecourse",
        "race_going",
        "distance_bucket",
    ]
    history = build_entity_history(jockey_results, "jockey_name", columns).rename(
        columns={
            "position_rank": "jockey_last_position_rank",
            "finish_position": "jockey_last_finish_position",
            "DNF": "jockey_last_DNF",
            "NR": "jockey_last_NR",
            "WD": "jockey_last_WD",
            "finish_time_sec": "jockey_last_finish_time_sec",
            "starting_prob": "jockey_last_starting_prob",
            "racecourse": "jockey_last_racecourse",
            "race_going": "jockey_last_race_going",
            "distance_bucket": "jockey_last_distance_bucket",
        },
    )
    return history.sort_values(["jockey_name", "record_date"], kind="stable").reset_index(drop=True)


def main() -> None:
    """Build all preprocessing outputs."""
    race_results = build_race_results()
    non_runner_events = build_non_runner_events()
    historical_features = build_historical_features(race_results)
    horse_history_df = build_horse_history_df(race_results)
    jockey_history_df = build_jockey_history_df(race_results)
    sequential_model_input = build_sequential_model_input(race_results, non_runner_events)

    race_results.to_csv(RACE_RESULTS_CSV, index=False)
    non_runner_events.to_csv(NON_RUNNER_EVENTS_CSV, index=False)
    historical_features.to_csv(HISTORICAL_FEATURES_CSV, index=False)
    horse_history_df.set_index(["horse_name", "record_date"]).to_csv(HORSE_HISTORY_CSV)
    jockey_history_df.set_index(["jockey_name", "record_date"]).to_csv(JOCKEY_HISTORY_CSV)
    MODEL_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sequential_model_input.to_csv(SEQUENTIAL_MODEL_INPUT_CSV, index=False)


if __name__ == "__main__":
    main()
