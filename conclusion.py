# %%
"""Analysis of Hierarchical Bayesian horse racing model results."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "model"

# %%
# Load saved posterior
idata = az.from_netcdf(MODEL_DIR / "model_fit.nc")


# Quick diagnostic summary
logger.info("Model summary:\n{}", az.summary(idata, var_names=["intercept", "sigma"]))


# %%
# Trace plots
az.plot_trace(idata, var_names=["intercept", "sigma"])
plt.show()

# %%
# Energy plot (to check for divergences / funnel issues)
az.plot_energy(idata)
plt.show()

# %%
# Rank plots to check mixing
az.plot_rank(idata, var_names=["intercept", "sigma"])
plt.show()

# %%
# Autocorrelation for parameters
az.plot_autocorr(idata, var_names=["intercept"])
plt.show()

# %%
# Load predictions
pred_df = pd.read_csv(OUTPUT_DIR / "predictions" / "predictions.csv")

# Check head
pred_df.head()

# %%
# Accuracy: Winner prediction
winners = pred_df.loc[pred_df.groupby("race_id")["pred_speed"].idxmax()]
actual_winners = pred_df.loc[pred_df.groupby("race_id")["speed"].idxmax()]
acc = accuracy_score(actual_winners["horse_name"], winners["horse_name"])
logger.info("Winner prediction accuracy: {:.2%}", acc)

# %%
# Top-3 accuracy
top3_acc = float(
    np.mean(
        [
            grp.loc[grp["speed"].idxmax()]["horse_name"]
            in grp.sort_values("pred_speed", ascending=False)["horse_name"].head(3).to_numpy()
            for _, grp in pred_df.groupby("race_id")
        ],
    ),
)
logger.info("Top-3 accuracy: {:.2%}", top3_acc)


# %%
# Classification view: how often predicted horse placed in actual race
def placing_accuracy(df: pd.DataFrame, k: int = 3) -> float:
    """Return fraction of races where actual winner was in top-k predicted horses."""
    return float(
        np.mean(
            [
                grp.loc[grp["speed"].idxmax()]["horse_name"]
                in grp.sort_values("pred_speed", ascending=False)["horse_name"].head(k).to_numpy()
                for _, grp in df.groupby("race_id")
            ],
        ),
    )


for k in [1, 2, 3, 5]:
    logger.info("Top-{} accuracy: {:.2%}", k, placing_accuracy(pred_df, k))

# %%
# Confusion matrix (only works well if relatively few horses; otherwise too big)
# We'll do predicted winner vs actual winner at the horse_name level
cm = confusion_matrix(actual_winners["horse_name"], winners["horse_name"], labels=None)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
plt.title("Confusion Matrix (Predicted vs Actual Winners)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# Distribution of predicted ranks for actual winners
actual_ranks = pred_df.loc[pred_df["won"], ["race_id", "pred_rank"]]["pred_rank"].values

plt.figure(figsize=(6, 4))
sns.histplot(actual_ranks, bins=range(1, 8), discrete=True)
plt.xlabel("Predicted rank of actual winner")
plt.ylabel("Count")
plt.title("How often did we rank the actual winner correctly?")
plt.show()

# %%
# ROI analysis
summary = pd.read_csv(OUTPUT_DIR / "predictions" / "race_summary.csv")

total_stakes = len(summary)
total_returns = summary["pred_winner_return"].sum()
roi = (total_returns - total_stakes) / total_stakes
logger.info("Total ROI from betting 1 unit per race: {:.2%}", roi)

plt.figure(figsize=(6, 4))
sns.histplot(summary["pred_winner_return"], bins=20)
plt.xlabel("Bet return per race")
plt.ylabel("Count")
plt.title("Distribution of per-race returns")
plt.show()
