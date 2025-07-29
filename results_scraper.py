# %%
"""Scrape British Horseracing Association results by month into JSON files."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

# %%
# --- Path & driver setup ---
BASE_DIR = Path(__file__).resolve().parent

CHROME_PATH = BASE_DIR / "chrome" / "chrome-win64" / "chrome-win64" / "chrome.exe"
CHROMEDRIVER_PATH = (
    BASE_DIR / "chromedriver" / "chromedriver-win64" / "chromedriver-win64" / "chromedriver.exe"
)

options = Options()
options.binary_location = str(CHROME_PATH)
options.add_argument("--start-maximized")
options.add_argument("--headless")

service = Service(executable_path=str(CHROMEDRIVER_PATH))
driver = webdriver.Chrome(service=service, options=options)


def safe_text(xpath: str) -> str | None:
    """Safely retrieve text content of the first element matching the given XPath."""
    try:
        element = driver.find_element(By.XPATH, xpath)
        return element.text.strip()
    except NoSuchElementException:
        return None


output_dir = BASE_DIR / "output"
results_dir = output_dir / "results"

results_dir.mkdir(parents=True, exist_ok=True)

# %%
# --- Loop over year/month from Jun 2025 to Jul 2025 ---
periods = pd.date_range(start="2025-06-01", end="2025-07-01", freq="MS")

for dt in periods:
    year, month = dt.year, dt.month
    ym = f"{year}_{month:02d}"
    logger.info(f"Processing {ym}")

    # load month page and scroll
    url = f"https://www.britishhorseracing.com/racing/results/#!?year={year}&month={month}"
    driver.get(url)
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # find fixtures with results
    items = driver.find_elements(By.CSS_SELECTOR, "#results-list > li")
    fixtures = [
        (idx, item.find_element(By.TAG_NAME, "a").get_attribute("href"))
        for idx, item in enumerate(items, start=1)
        if item.find_elements(By.XPATH, ".//div[4]/span/span[contains(.,'View results')]")
    ]

    all_fixtures_data = []
    main_tab = driver.current_window_handle

    # scrape each fixture in its own tab
    for fixture_idx, fixture_url in fixtures:
        logger.info(f"Fixture {fixture_idx} out of {len(fixtures)}")
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get(fixture_url)

        wait = WebDriverWait(driver, 15)
        wait.until(ec.presence_of_element_located((By.ID, "racecard-ui")))

        racecourse = wait.until(
            ec.presence_of_element_located((By.XPATH, '//*[@id="fixture-header-ui"]/h2/div[1]')),
        ).text

        # expand all race cards
        for btn in driver.find_elements(By.XPATH, "//*[@id='racecard-ui']/li/div[1]/div[5]/span"):
            driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            time.sleep(0.1)
            btn.click()
            time.sleep(0.1)

        # collect races
        races = []
        raw_cards = driver.find_elements(By.CSS_SELECTOR, "#racecard-ui > li")

        # keep only those that actually contain a race-time cell
        cards = [
            c
            for c in raw_cards
            if c.find_elements(By.CSS_SELECTOR, "div.table-cell.w10.time span.inline-field-value")
        ]

        for i, card in enumerate(cards, start=1):
            race_time = card.find_element(By.CSS_SELECTOR, ".table-cell.w10.time").text
            race_name = card.find_element(By.CSS_SELECTOR, ".table-cell.w40.name").text
            race_dist = card.find_element(By.CSS_SELECTOR, ".table-cell.w20.race-distance").text
            winner = card.find_element(By.CSS_SELECTOR, ".table-cell.w25.last-col.winner").text

            logger.info(f"Fixture {fixture_idx}, Race {i} out of {len(cards)}, {race_name}")

            # runner-level info: capture each column's raw text
            runners = []
            for j, entry in enumerate(card.find_elements(By.XPATH, "./div[2]//ul/li"), start=1):
                cells = entry.find_elements(By.CSS_SELECTOR, "div.table-cell")
                cols = ["position", "horse_jockey", "trainer_owner", "distance_time", "sp"]
                data = {"runner_index": j}
                for idx, cell in enumerate(cells):
                    key = cols[idx] if idx < len(cols) else f"col_{idx}"
                    data[key] = cell.text.strip()
                runners.append(data)

            races.append(
                {
                    "race_index": i,
                    "racecourse": racecourse,
                    "time": race_time,
                    "name": race_name,
                    "distance": race_dist,
                    "winner_info": winner,
                    "runners": runners,
                },
            )

        all_fixtures_data.append(
            {
                "fixture_index": fixture_idx,
                "fixture_url": fixture_url,
                "date": safe_text("//*[@id='fixture-header-ui']/h2/div[2]/span[1]"),
                "year": year,
                "races": races,
            },
        )

        driver.close()
        driver.switch_to.window(main_tab)

    driver.close()
    # write JSON for this month
    out_file = results_dir / f"{ym}.json"
    with out_file.open("w") as f:
        json.dump(all_fixtures_data, f, indent=2)

    logger.info(f"Wrote {len(all_fixtures_data)} fixtures to {out_file}")


# %%
# ! Formatting Outputs


# ? Helper Funcs
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
# Collect flattened records from the new JSON layout using list.extend
records = []
for file in sorted(results_dir.glob("*.json")):
    ym = file.stem  # e.g. "2025_06"
    with file.open() as f:
        fixtures = json.load(f)
    for fixture in fixtures:
        for race in fixture["races"]:
            for runner in race["runners"]:
                rec = {
                    "year_month": ym,
                    "fixture_index": fixture["fixture_index"],
                    "fixture_date": fixture["date"],
                    "fixture_year": fixture["year"],
                    "race_index": race["race_index"],
                    "racecourse": race["racecourse"],
                    "race_time_raw": race.get("time"),
                    "race_name_raw": race.get("name"),
                    "race_distance_raw": race.get("distance"),
                    "winner_info_raw": race.get("winner_info"),
                    "runner_index": runner.get("runner_index"),
                    "position_raw": runner.get("position"),
                    "horse_jockey_raw": runner.get("horse_jockey"),
                    "trainer_owner_raw": runner.get("trainer_owner"),
                    "distance_time_raw": runner.get("distance_time"),
                    "sp_raw": runner.get("sp"),
                }
                # Parse position
                if rec["position_raw"]:
                    parts = rec["position_raw"].split("\n")
                    rec["position_rank"] = parts[0]
                    rec["position_extra"] = parts[1] if len(parts) > 1 else None
                else:
                    rec["position_rank"] = None
                    rec["position_extra"] = None

                # Parse horse & jockey block
                hj = rec["horse_jockey_raw"] or ""
                lines = hj.split("\n")
                rec["horse_name"] = lines[0] if len(lines) > 0 else None
                rec["jockey_name"] = lines[1] if len(lines) > 1 else None
                for line in lines[2:]:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        key = key.strip().lower().replace(" ", "_")
                        rec[f"{key}_raw"] = val.strip()

                # Parse trainer & owner
                to = rec["trainer_owner_raw"] or ""
                tlines = to.split("\n")
                rec["trainer_name"] = tlines[0] if len(tlines) > 0 else None
                rec["owner_name"] = tlines[1] if len(tlines) > 1 else None

                # Parse distance & finish time
                dt = rec["distance_time_raw"] or ""
                dt_lines = dt.split("\n") if dt else []
                if len(dt_lines) == 2:
                    rec["distance_raw"] = dt_lines[0]
                    rec["finish_time_raw"] = dt_lines[1]
                elif len(dt_lines) == 1:
                    # if no length, assume this is time
                    text = dt_lines[0]
                    if re.search(r"\d+[ms]", text):
                        rec["finish_time_raw"] = text
                        rec["distance_raw"] = None
                    else:
                        rec["distance_raw"] = text
                        rec["finish_time_raw"] = None
                else:
                    rec["distance_raw"] = None
                    rec["finish_time_raw"] = None

                # SP
                rec["sp"] = rec["sp_raw"]

                records.append(rec)

# Create DataFrame
results_raw = pd.DataFrame(records)

raw_columns = [c for c in results_raw.columns if "raw" in c]

results_df = results_raw.drop(columns=raw_columns)

results_df.to_csv(results_dir / "results_formatted.csv")
# %%
