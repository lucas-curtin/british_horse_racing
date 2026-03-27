# %%
"""Analysis of Hierarchical Bayesian horse racing model results with convergence."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

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

# BFMI

logger.info("BFMI:{}", az.bfmi(idata))

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


# Posterior distributions (intercept and sigma)
az.plot_posterior(idata, var_names=["intercept", "sigma"], hdi_prob=0.95)
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
