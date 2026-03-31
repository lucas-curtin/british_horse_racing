<p align="center">
  <img src="logo.png" alt="Project logo" width="200">
</p>

# British Horse Racing - Scraper and Bayesian Model

This repository currently contains a historical-results pipeline for British horse racing:

- `full_scraper.py` scrapes monthly BHA results and fixture-level non-runner information
- `preprocessing.py` converts the scraped JSON into wide audit tables plus curated model-input CSVs
- `temp_model.py` and `model.py` fit PyMC models and evaluate predictions on a held-out historical split
- `sequential_model.py` validates and loads the explicit feature-contract input for the next sequential model path

Important: the checked-in code is not currently a clean "tomorrow fixtures in, tomorrow predictions out" pipeline. The current modelling flow trains on historical results and scores a held-out test set from those same historical data.

## Current Setup

### 1. Create and activate the Conda environment

Create the environment once:

```bash
conda env update -f environment.yml --prune
```

Activate it for each session:

```bash
conda activate british_horse_racing
```

### 2. Check the browser paths in `config.yml`

The scraper reads its browser paths from [config.yml](/Users/lucascurtin/Desktop/repos/british_horse_racing/config.yml).

Current defaults on this Mac setup:

- `chromedriver_path` points to `chromedriver-mac-x64/chromedriver`
- `chrome_path` points to `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`

The scraper now falls back to Selenium's default Chrome discovery and Selenium Manager if those local paths are missing, so it is more forgiving than before. If you want to set a custom Chrome binary explicitly, use:

```yaml
chrome_path: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
```

Apple Silicon note: this repo's checked-in `chromedriver-mac-x64/chromedriver` is an Intel (`x86_64`) binary. On an `arm64` Mac, Selenium Manager is likely the safer path unless you know that driver works under Rosetta with your local Chrome install.

### 3. Set the scrape date range

Still in [config.yml](/Users/lucascurtin/Desktop/repos/british_horse_racing/config.yml), set:

```yaml
dates:
  start: "2026-02-27"
  end: "2026-03-27"
```

These are the current defaults and represent the last few days relative to March 27, 2026. The scraper uses monthly result pages, but it filters fixtures by their actual race date. Treat both `start` and `end` as inclusive. For a single-day run, set them to the same date.

## Run Order

### 1. Scrape fixture results

```bash
python full_scraper.py
```

This script:

- scrapes monthly BHA race results into `output/results/*.json`
- captures fixture-level non-runner entries alongside race results
- records unresolved fixtures in `output/missed/*.json` and retries them once at the end of the run

### 2. Build the modelling datasets

```bash
python preprocessing.py
```

This writes:

- `output/race_results.csv`
- `output/non_runner_events.csv`
- `output/historical_features.csv`
- `output/horse_df.csv`
- `output/jockey_df.csv`
- `output/model_inputs/sequential_ranking_input.csv`

### 3. Train the model and generate evaluation predictions

Recommended current model entrypoint:

```bash
python temp_model.py
```

This writes:

- `output/model/model_fit.nc`
- `output/predictions/predictions.csv`
- `output/predictions/race_summary.csv`

There is also an older alternative:

```bash
python model.py
```

## What Each Model Script Does

- `temp_model.py`: the safer current option; trains from `output/historical_features.csv`, handles unseen categories more defensively, and evaluates on a held-out chronological split
- `model.py`: similar historical train/test evaluation flow, but less defensive than `temp_model.py`

## Expected Outputs

After a successful historical run, you should have:

- [output/results](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/results)
- [output/race_results.csv](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/race_results.csv)
- [output/non_runner_events.csv](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/non_runner_events.csv)
- [output/historical_features.csv](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/historical_features.csv)
- [output/model_inputs/sequential_ranking_input.csv](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/model_inputs/sequential_ranking_input.csv)
- [output/model/model_fit.nc](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/model/model_fit.nc)
- [output/predictions/predictions.csv](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/predictions/predictions.csv)
- [output/predictions/race_summary.csv](/Users/lucascurtin/Desktop/repos/british_horse_racing/output/predictions/race_summary.csv)

## Known Gaps

- The README that originally shipped with the repo referred to scripts like `results_scraper.py`, `fixture_scraper.py`, `horse_jockey_scraper.py`, and `bayesian.py`, but those files are not present in the current checkout.
- The current pipeline is for historical scraping and evaluation, not next-day fixture prediction.
- On Apple Silicon, you may still want to rely on Selenium Manager rather than the checked-in Intel chromedriver binary.

## Quick Start

Once the environment is active:

```bash
python full_scraper.py
python preprocessing.py
python temp_model.py
```
