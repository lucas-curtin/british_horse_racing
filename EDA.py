# %%
"""EDA Script."""

from pathlib import Path

import pandas as pd

# --- File path constants ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"


race_df = pd.read_csv(OUTPUT_DIR / "race_df.csv")
jockey_df = pd.read_csv(OUTPUT_DIR / "jockey_df.csv")
horse_df = pd.read_csv(OUTPUT_DIR / "horse_df.csv")

# %%
# ? Combining dataframes
merged_df = race_df.merge(horse_df, on="horse_name", how="left").merge(
    jockey_df,
    on="jockey_name",
    how="left",
)

merged_df["race_age"] = (merged_df["race_date"] - merged_df["foaled"]).dt.days


merged_df.set_index("race_date").to_csv(OUTPUT_DIR / "merged_results.csv")
# %%
