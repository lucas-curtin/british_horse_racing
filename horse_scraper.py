"""Grab horse information from fixture list."""

# %%
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

# %%
# ? Setting up params
output_dir = Path("output")
results_dir = output_dir / "results"


# %%
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

# %%
# ? Extracting Horses Names

unique_horses = results_df["horse_name"].unique()
no_bracket_horses = [re.sub(r"\s*\([^)]*\)$", "", h) for h in unique_horses]


# %%
# ? Opening Page
BASE_DIR = Path(__file__).resolve().parent
chrome_bin = (
    BASE_DIR
    / "chrome-mac-x64"
    / "Google Chrome for Testing.app"
    / "Contents"
    / "MacOS"
    / "Google Chrome for Testing"
)
chromedriver_path = BASE_DIR / "chromedriver-mac-x64" / "chromedriver"
chrome_bin = str(chrome_bin)
chromedriver_path = str(chromedriver_path)

options = Options()
options.binary_location = chrome_bin
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1200")

service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service, options=options)

base_url = "https://www.britishhorseracing.com/racing/horses/"
driver.get(base_url)


# %%
# ? Grabbing Horse Details

horse_details = []
main_tab = driver.current_window_handle
wait = WebDriverWait(driver, 15)

for horse in unique_horses:
    # 1) Search the horse
    search_input = wait.until(
        ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[1]/div[2]/input')),
    )
    search_input.clear()
    search_input.send_keys(horse)

    go_button = wait.until(ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[2]')))
    go_button.click()

    # 2) Open the first result
    first_link = wait.until(ec.element_to_be_clickable((By.XPATH, '//*[@id="horses-list"]/div/a')))
    href = first_link.get_attribute("href")
    driver.execute_script("window.open(arguments[0]);", href)
    driver.switch_to.window(driver.window_handles[-1])

    # 3) Scrape the general and specific info blocks
    general = wait.until(
        ec.presence_of_element_located((By.XPATH, '//*[@id="horse-single-info"]/div[1]')),
    ).text
    specific = driver.find_element(By.XPATH, '//*[@id="horse-single-info"]/div[2]').text

    # 4) Build a dict with the raw text
    data = {
        "name": horse,
    }

    lines = general.split("\n")

    # if there's a second line, extract type and birth_year
    if len(lines) > 1:
        desc = lines[1]
        # split into ["BAY GELDING ", "2022"]
        type_part, year_part = desc.split("b.", 1)
        data["type"] = type_part.strip()  # => "BAY GELDING"
        data["birth_year"] = year_part.strip()  # => "2022"

    for line in specific.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        if key == "associated_content":
            continue
        data[key] = val.strip()

    horse_details.append(data)

    # 5) Close detail tab and switch back
    driver.close()
    driver.switch_to.window(main_tab)


# %%
# ? Save the data
with (output_dir / "horse_details.json").open("w", encoding="utf-8") as f:
    json.dump(horse_details, f, ensure_ascii=False, indent=4)

# %%
