# %%
"""Model Script."""

from pathlib import Path

import pandas as pd

# --- File path constants ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"


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

race_raw = pd.read_csv(OUTPUT_DIR / "race_df.csv")

flat_df = (
    race_raw[race_raw[race_types].any(axis=1)]
    .loc[(~race_raw["WD"]) & (~race_raw["NR"]) & (~race_raw["DNF"])]
    .assign(speed=lambda df: df["race_distance_m"] / df["finish_time_sec"])
)


flat_slim = flat_df[
    [
        "fixture_index",
        "race_index",
        "racecourse",
        "going_stick",
        "soil_moisture_pct",
        # "runner_index",
        # "position_rank",
        "draw_number",
        "horse_name",
        "jockey_name",
        "handicap_ran_off",
        # "bha_performance_figure",
        "current_mark_surface",
        "current_mark",
        "trainer_name",
        "owner_name",
        # "race_distance_m",
        # "finish_time_sec",
        # "race_date",
        # "starting_prob",
        # "won",
        "weather_category",
        # "DNF",
        # "WD",
        # "NR",
        # "handicap",
        # "steeple",
        # "chase",
        # "novice",
        # "hurdle",
        # "maiden",
        # "national_hunt",
        # "selling",
        "class",
    ]
]


# %%
# ? Combining dataframes

# %%
