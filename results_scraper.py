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

        going_text = wait.until(
            ec.presence_of_element_located(
                (By.XPATH, '//*[@id="fixture-header-ui"]/div/div/div[1]')
            )
        ).text

        # Weather
        weather_text = wait.until(
            ec.presence_of_element_located(
                (By.XPATH, '//*[@id="fixture-header-ui"]/div/div/div[2]')
            )
        ).text

        # Other info
        other_text = wait.until(
            ec.presence_of_element_located(
                (By.XPATH, '//*[@id="fixture-header-ui"]/div/div/div[3]')
            )
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
                "racecourse": racecourse,
                "going": going_text,
                "weather": weather_text,
                "other_text": other_text,
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


# Constants for conversion
MILE_M = 1609.34  # metres per mile
FURLONG_M = 201.168  # metres per furlong
YARD_M = 0.9144  # metres per yard

records = []
for file in sorted(results_dir.glob("*.json")):
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
                    # "going": fixture["going"],
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
                rec["position_rank"] = parts[0] if parts else ""
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
raw_cols = [c for c in results_raw.columns if c.endswith("_raw")]
results_df = (
    (
        results_raw.drop(columns=raw_cols)
        .set_index("year_month")
        .drop(columns=["position_extra", "race_distance"])
    )
    .astype({"fixture_date": "str", "fixture_year": "str"})
    .assign(race_date=lambda df: df["fixture_date"] + " " + df["fixture_year"])
)
results_df.to_csv(results_dir / "results_formatted.csv")

# %%
