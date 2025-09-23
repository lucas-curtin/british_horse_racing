# %%
"""Analysis of Hierarchical Bayesian horse racing model results with convergence."""

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

# ---------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------
summary = az.summary(idata, var_names=["intercept", "sigma"], round_to=2)
logger.info("Model summary:\n{}", summary)

# R-hat and ESS
logger.info("Max R-hat: {:.3f}", summary["r_hat"].max())
logger.info("Min ESS: {:.0f}", summary["ess_bulk"].min())

# Divergences
n_div = idata.sample_stats["diverging"].sum().item()
logger.info("Number of divergences: {}", n_div)

# %%
# Trace plots
az.plot_trace(idata, var_names=["intercept", "sigma"])
plt.show()

# Energy plot (funnel issues)
az.plot_energy(idata)
plt.show()

# Rank plots to check mixing
az.plot_rank(idata, var_names=["intercept", "sigma"])
plt.show()

# Autocorrelation
az.plot_autocorr(idata, var_names=["intercept"])
plt.show()

# %%
# Posterior distributions (intercept and sigma)
az.plot_posterior(idata, var_names=["intercept", "sigma"], hdi_prob=0.95)
plt.show()

# %%
# Posterior predictive checks
ppc = az.from_netcdf(MODEL_DIR / "model_fit.nc")  # reload if needed
az.plot_ppc(idata, data_pairs={"obs": "obs"})
plt.show()

# ---------------------------------------------------------------------
# Prediction analysis
# ---------------------------------------------------------------------
pred_df = pd.read_csv(OUTPUT_DIR / "predictions" / "predictions.csv")

winners = pred_df.loc[pred_df.groupby("race_id")["pred_speed"].idxmax()]
actual_winners = pred_df.loc[pred_df.groupby("race_id")["speed"].idxmax()]
acc = accuracy_score(actual_winners["horse_name"], winners["horse_name"])
logger.info("Winner prediction accuracy: {:.2%}", acc)


def placing_accuracy(df: pd.DataFrame, k: int = 3) -> float:
    """Accuracy in palcements."""
    return float(
        np.mean(
            [
                grp.loc[grp["speed"].idxmax(), "horse_name"]
                in grp.sort_values("pred_speed", ascending=False)["horse_name"].head(k).to_numpy()
                for _, grp in df.groupby("race_id")
            ],
        ),
    )


for k in [1, 2, 3, 5]:
    logger.info("Top-{} accuracy: {:.2%}", k, placing_accuracy(pred_df, k))

# Confusion matrix
cm = confusion_matrix(actual_winners["horse_name"], winners["horse_name"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, cmap="Blues", cbar=True)
plt.title("Confusion Matrix (Predicted vs Actual Winners)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# ROI analysis
summary = pd.read_csv(OUTPUT_DIR / "predictions" / "race_summary.csv")

roi = (summary["pred_winner_return"].sum() - len(summary)) / len(summary)
logger.info("Total ROI from betting 1 unit per race: {:.2%}", roi)

plt.figure(figsize=(6, 4))
sns.histplot(summary["pred_winner_return"], bins=20, kde=False)
plt.xlabel("Bet return per race")
plt.ylabel("Count")
plt.title("Distribution of per-race returns")
plt.show()

# %%
# Group-level effects (forest plots)
for var in [v for v in idata.posterior.data_vars if v.endswith("_eff")]:
    az.plot_forest(idata, var_names=[var], combined=True, hdi_prob=0.95, r_hat=True)
    plt.title(f"Posterior of {var}")
    plt.show()

# Optional: look at top/bottom effects (e.g. horses with strongest signals)
if "horse_name_eff" in idata.posterior:
    eff = idata.posterior["horse_name_eff"].mean(("chain", "draw")).values  # noqa: PD011
    top_idx = np.argsort(eff)[-10:]
    bottom_idx = np.argsort(eff)[:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=np.concatenate([eff[bottom_idx], eff[top_idx]]),
        y=np.concatenate([bottom_idx, top_idx]),
    )
    plt.title("Top and Bottom Horse Effects (posterior means)")
    plt.xlabel("Effect on log-speed")
    plt.show()
