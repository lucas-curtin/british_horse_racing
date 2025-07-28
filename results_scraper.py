# %%
"""Scrape British Horseracing Association results by month into JSON files."""

from __future__ import annotations

import json
import time
from pathlib import Path

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


def safe_text(xpath: str) -> str | None:
    """Safely retrieve the text content of the first element matching the given XPath."""
    try:
        return driver.find_element(By.XPATH, xpath).text.strip()
    except NoSuchElementException:
        return None


# ensure output directory exists
output_dir = BASE_DIR / "output" / "results"
output_dir.mkdir(parents=True, exist_ok=True)

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
            # race-level info (you can handle parsing later)

            race_time = card.find_element(By.CSS_SELECTOR, ".table-cell.w10.time").text
            race_name = card.find_element(By.CSS_SELECTOR, ".table-cell.w40.name").text
            race_dist = card.find_element(By.CSS_SELECTOR, ".table-cell.w20.race-distance").text
            winner = card.find_element(By.CSS_SELECTOR, ".table-cell.w25.last-col.winner").text

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
        break

    driver.close()
    # write JSON for this month
    out_file = output_dir / f"{ym}.json"
    with out_file.open("w") as f:
        json.dump(all_fixtures_data, f, indent=2)

    logger.info(f"Wrote {len(all_fixtures_data)} fixtures to {out_file}")


# %%
