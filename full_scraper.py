#!/usr/bin/env python3
"""Combined script: scrape BHA results by month, then scrape horse and jockey details with detailed logging."""

# %%
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

# ————— Global paths & settings —————
BASE_DIR = Path(__file__).resolve().parent
CHROME_PATH = BASE_DIR / "chrome" / "chrome-win64" / "chrome-win64" / "chrome.exe"
CHROMEDRIVER_PATH = (
    BASE_DIR / "chromedriver" / "chromedriver-win64" / "chromedriver-win64" / "chromedriver.exe"
)
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ————— WebDriver factory —————
def make_driver() -> webdriver.Chrome:
    """
    Create and configure a headless Chrome WebDriver instance.

    Returns:
        webdriver.Chrome: Configured headless Chrome WebDriver.
    """
    options = Options()
    options.binary_location = str(CHROME_PATH)
    options.add_argument("--start-maximized")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    service = Service(executable_path=str(CHROMEDRIVER_PATH), log_path="NUL")
    return webdriver.Chrome(service=service, options=options)


# ————— Utility —————
def safe_text(driver: webdriver.Chrome, xpath: str) -> str | None:
    """
    Safely retrieve the text content of the first element matching the given XPath.

    Args:
        driver (webdriver.Chrome): The WebDriver instance.
        xpath (str): The XPath string to locate the element.

    Returns:
       Optional[str]: Stripped text of the element, or None if not found.
    """
    try:
        elem = driver.find_element(By.XPATH, xpath)
        return elem.text.strip()
    except NoSuchElementException:
        return None


# ————— Part 1: Scrape monthly fixture results —————
def scrape_monthly_results(start: date, end: date) -> None:
    """
    Scrape BHA race results for each month between start and end dates, store as JSON files.

    Args:
        start (date): The first day of the starting month (inclusive).
        end (date): The first day of the month after the final month (exclusive).

    Returns:
        None: Writes JSON files to RESULTS_DIR.
    """
    driver = make_driver()
    periods = pd.date_range(start=start, end=end, freq="MS")

    for dt in periods:
        year, month = dt.year, dt.month
        ym = f"{year}_{month:02d}"
        logger.info(f"Processing month: {ym}")

        url = f"https://www.britishhorseracing.com/racing/results/#!?year={year}&month={month}"
        driver.get(url)
        driver.refresh()
        time.sleep(1)

        # Wait for the results list container to appear
        results_list = WebDriverWait(driver, 10).until(
            ec.presence_of_element_located((By.ID, "results-list"))
        )

        # Infinite scroll: collect unique fixture URLs
        seen_hrefs = set()
        while True:
            all_items = results_list.find_elements(By.TAG_NAME, "li")
            fixture_items = [li for li in all_items if li.find_elements(By.TAG_NAME, "a")]
            hrefs = [
                li.find_element(By.TAG_NAME, "a").get_attribute("href") for li in fixture_items
            ]

            # identify new hrefs
            new_hrefs = [href for href in hrefs if href not in seen_hrefs]
            if not new_hrefs:
                break

            seen_hrefs.update(new_hrefs)
            logger.info(f"Scrolled: total unique fixtures so far = {len(seen_hrefs)}")

            # scroll last fixture into view to trigger load
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'end', behavior: 'smooth'});",
                fixture_items[-1],
            )
            time.sleep(2)

        # Build fixtures list preserving first-seen order
        fixtures = [(idx + 1, href) for idx, href in enumerate(seen_hrefs)]

        all_fixtures_data: list[dict] = []
        main_tab = driver.current_window_handle

        for fixture_idx, fixture_url in fixtures:
            logger.info(f"Fixture {fixture_idx}/{len(fixtures)}: opening {fixture_url}")
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[-1])
            driver.get(fixture_url)

            wait = WebDriverWait(driver, 15)
            try:
                wait.until(ec.presence_of_element_located((By.ID, "racecard-ui")))
            except TimeoutException:
                logger.warning(f"No race data for fixture {fixture_url}, skipping.")
                driver.close()
                driver.switch_to.window(main_tab)
                continue

            racecourse = wait.until(
                ec.presence_of_element_located((By.XPATH, '//*[@id="fixture-header-ui"]/h2/div[1]'))
            ).text
            going_text = safe_text(driver, '//*[@id="fixture-header-ui"]/div/div/div[1]')
            weather_text = safe_text(driver, '//*[@id="fixture-header-ui"]/div/div/div[2]')
            other_text = safe_text(driver, '//*[@id="fixture-header-ui"]/div/div/div[3]')

            # Expand all race cards
            for btn in driver.find_elements(
                By.XPATH, "//*[@id='racecard-ui']/li/div[1]/div[5]/span"
            ):
                driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                time.sleep(0.1)
                btn.click()
                time.sleep(0.1)

            raw_cards = driver.find_elements(By.CSS_SELECTOR, "#racecard-ui > li")
            cards = [
                c
                for c in raw_cards
                if c.find_elements(
                    By.CSS_SELECTOR, "div.table-cell.w10.time span.inline-field-value"
                )
            ][:1]

            races: list[dict] = []
            for i, card in enumerate(cards, start=1):
                race_time = card.find_element(By.CSS_SELECTOR, ".table-cell.w10.time").text
                race_name = card.find_element(By.CSS_SELECTOR, ".table-cell.w40.name").text
                race_dist = card.find_element(By.CSS_SELECTOR, ".table-cell.w20.race-distance").text
                winner = card.find_element(By.CSS_SELECTOR, ".table-cell.w25.last-col.winner").text

                runners: list[dict] = []
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
                    }
                )

            all_fixtures_data.append(
                {
                    "fixture_index": fixture_idx,
                    "fixture_url": fixture_url,
                    "date": safe_text(driver, '//*[@id="fixture-header-ui"]/h2/div[2]/span[1]'),
                    "year": year,
                    "racecourse": racecourse,
                    "going": going_text,
                    "weather": weather_text,
                    "other_text": other_text,
                    "races": races,
                }
            )

            driver.close()
            driver.switch_to.window(main_tab)

        # write JSON for this month
        out_file = RESULTS_DIR / f"{ym}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_fixtures_data, f, indent=2)
        logger.info(f"Wrote {len(all_fixtures_data)} fixtures to {out_file}")

    driver.quit()


# ————— Part 2: Scrape horse and jockey details —————
def scrape_horses(horses_chunk: list[str], progress) -> list[dict]:
    """
    Scrape detailed information for each horse in the list.

    Args:
        horses_chunk (list[str]): Sublist of horse names to scrape.
        progress (tqdm): Progress bar instance to update.

    Returns:
        list[dict]: List of horse detail dictionaries.
    """
    driver = make_driver()
    wait = WebDriverWait(driver, 5)
    main_tab = driver.current_window_handle
    base_url = "https://www.britishhorseracing.com/racing/horses/"
    details: list[dict] = []

    for horse in horses_chunk:
        logger.info(f"Scraping horse profile: {horse}")
        try:
            driver.get(base_url)
            inp = wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[1]/div[2]/input'))
            )
            inp.clear()
            inp.send_keys(horse)
            wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[2]'))
            ).click()

            link = wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="horses-list"]/div/a'))
            )
            href = link.get_attribute("href")
            driver.execute_script("window.open(arguments[0]);", href)
            driver.switch_to.window(driver.window_handles[-1])

            general = wait.until(
                ec.presence_of_element_located((By.XPATH, '//*[@id="horse-single-info"]/div[1]'))
            ).text
            specific = driver.find_element(By.XPATH, '//*[@id="horse-single-info"]/div[2]').text

            data: dict = {"name": horse}
            lines = general.split("\n")
            if len(lines) > 1 and "b." in lines[1]:
                t, y = lines[1].split("b.", 1)
                data["type"] = t.strip()
                data["birth_year"] = y.strip()

            for line in specific.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip().lower().replace(" ", "_")
                if k != "associated_content":
                    data[k] = v.strip()

            details.append(data)
        except Exception as e:
            logger.error(f"Failed to scrape horse {horse}: {e}")
        finally:
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(main_tab)
            progress.update(1)

    driver.quit()
    return details


def scrape_jockeys(jockeys_chunk: list[str], progress) -> list[dict]:
    """
    Scrape detailed information for each jockey in the list.

    Args:
        jockeys_chunk (list[str]): Sublist of jockey names to scrape.
        progress (tqdm): Progress bar instance to update.

    Returns:
        list[dict]: List of jockey detail dictionaries.
    """
    driver = make_driver()
    wait = WebDriverWait(driver, 15)
    base_url = "https://www.britishhorseracing.com/racing/participants/jockeys/"
    details: list[dict] = []

    for jockey in jockeys_chunk:
        logger.info(f"Scraping jockey profile: {jockey}")
        try:
            driver.get(base_url)
            time.sleep(10)
            inp = wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[1]//input'))
            )
            inp.clear()
            inp.send_keys(jockey)
            wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[2]'))
            ).click()

            wait.until(ec.presence_of_element_located((By.XPATH, '//*[@id="jockeys-list"]/div')))
            driver.find_element(By.XPATH, '//*[@id="jockeys-list"]/div').click()

            info = wait.until(
                ec.presence_of_element_located(
                    (By.XPATH, '//*[@id="jockey-single-info"]/div[2]/div[2]')
                )
            ).text
            career = driver.find_element(
                By.XPATH, '//*[@id="jockey-single-info"]/div[2]/div[3]'
            ).text

            details.append({"name": jockey, "jockey_info": info, "career_info": career})
        except Exception as e:
            logger.error(f"Failed to scrape jockey {jockey}: {e}")
        finally:
            progress.update(1)

    driver.quit()
    return details


def run_stage(name: str, items: list[str], scrape_fn, out_file: str, num_workers: int = 8) -> None:
    """
    Generic runner to scrape details in parallel and save to JSON.

    Args:
        name (str): Stage name for logging and progress bar.
        items (list[str]): List of items (horse/jockey names).
        scrape_fn: Function to scrape a list chunk.
        out_file (str): Output filename under OUTPUT_DIR.
        num_workers (int): Number of parallel threads.

    Returns:
        None: Writes combined results to OUTPUT_DIR/out_file.
    """
    chunk_size = (len(items) + num_workers - 1) // num_workers
    chunks = [items[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)]

    results: list[dict] = []
    with tqdm(total=len(items), desc=f"{name} Progress") as progress:
        with ThreadPoolExecutor(max_workers=num_workers) as exe:
            futures = [exe.submit(scrape_fn, chunk, progress) for chunk in chunks]
            for fut in as_completed(futures):
                results.extend(fut.result())

    path = OUTPUT_DIR / out_file
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.success(f"Wrote {len(results)} {name.lower()} details to {path}")


# ————— Main runner —————
def main() -> None:
    """
    Execute monthly scraping, aggregate runner names, then scrape horse and jockey details.

    Returns:
        None
    """
    # Scrape monthly results (adjust dates as needed)
    scrape_monthly_results(start=date(2025, 7, 1), end=date(2025, 7, 1))

    # Aggregate unique horse/jockey names
    all_results: list[dict] = []
    for file in RESULTS_DIR.glob("*.json"):
        with file.open("r", encoding="utf-8") as f:
            all_results.extend(json.load(f))

    horse_list = []
    jockey_list = []

    for fixture in all_results:
        # each fixture has a list of races
        for race in fixture.get("races", []):
            # now grab each runner in that race
            for runner in race.get("runners", []):
                parts = [
                    line.strip()
                    for line in runner.get("horse_jockey", "").splitlines()
                    if line.strip()
                ]
                if not parts:
                    continue

                # line 0 is the horse
                horse_list.append(parts[0])

                # line 1, if present and not "Non Runner", is the jockey
                if len(parts) > 1 and parts[1] != "Non Runner":
                    jockey_list.append(parts[1])

    # finally, dedupe
    horse_names = set(horse_list)
    jockey_names = set(jockey_list)

    # Scrape details in parallel
    run_stage("Jockey", sorted(jockey_names), scrape_jockeys, "jockey_details.json", num_workers=8)
    run_stage("Horse", sorted(horse_names), scrape_horses, "horse_details.json", num_workers=8)


# %%
if __name__ == "__main__":
    main()
