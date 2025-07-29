#!/usr/bin/env python3
"""Scrape horse and jockey information from fixture list, with parallelism and progress bars."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from selenium import webdriver
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

# Read the fixture results
results_df = pd.read_csv(RESULTS_DIR / "results_formatted.csv")

# Prepare horse list
unique_horses = results_df["horse_name"].unique().tolist()
unique_horses = [re.sub(r"\s*\([^)]*\)$", "", h) for h in unique_horses]

# Prepare jockey list
unique_jockeys = results_df["jockey_name"].unique().tolist()
unique_jockeys = [j for j in unique_jockeys if j != "Non Runner"]


def make_driver(headless: bool = True):
    options = Options()
    options.binary_location = str(CHROME_PATH)
    options.add_argument("--start-maximized")
    if headless:
        options.add_argument("--headless")
    # disable sandbox and GPU to avoid Windows permission errors
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    # suppress unnecessary logs
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    service = Service(executable_path=str(CHROMEDRIVER_PATH), log_path="NUL")
    return webdriver.Chrome(service=service, options=options)


def scrape_horses(horses_chunk, progress):
    """
    Spins up its own Chrome, scrapes a batch of horses, updates `progress`,
    returns list of dicts.
    """
    driver = make_driver(headless=True)
    wait = WebDriverWait(driver, 15)
    main_tab = driver.current_window_handle
    base_url = "https://www.britishhorseracing.com/racing/horses/"
    details = []

    for horse in horses_chunk:
        logger.info(f"[Horse] {horse}")
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

            data = {"name": horse}
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
            logger.error(f"[Horse] Failed {horse}: {e}")
        finally:
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(main_tab)
            progress.update(1)

    driver.quit()
    return details


def scrape_jockeys(jockeys_chunk, progress):
    """
    Spins up its own Chrome, scrapes a batch of jockeys, updates `progress`,
    returns list of dicts.
    """
    driver = make_driver(headless=True)
    wait = WebDriverWait(driver, 15)
    base_url = "https://www.britishhorseracing.com/racing/participants/jockeys/"
    details = []

    for jockey in jockeys_chunk:
        logger.info(f"[Jockey] {jockey}")
        try:
            driver.get(base_url)
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

            wait.until(
                ec.presence_of_element_located((By.XPATH, '//*[@id="jockey-single-info"]/div[2]'))
            )
            info = driver.find_element(By.XPATH, '//*[@id="jockey-single-info"]/div[2]/div[2]').text
            career = driver.find_element(
                By.XPATH, '//*[@id="jockey-single-info"]/div[2]/div[3]'
            ).text

            details.append(
                {
                    "name": jockey,
                    "jockey_info": info,
                    "career_info": career,
                }
            )

        except Exception as e:
            logger.error(f"[Jockey] Failed {jockey}: {e}")
        finally:
            progress.update(1)

    driver.quit()
    return details


def run_stage(name, items, scrape_fn, out_file, num_workers=8):
    """
    Generic runner: splits `items` into `num_workers` chunks,
    runs `scrape_fn` in parallel with a tqdm bar, saves to `out_file`.
    """
    chunk_size = (len(items) + num_workers - 1) // num_workers
    chunks = [items[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)]

    results = []
    with tqdm(total=len(items), desc=f"{name} Progress") as progress:
        with ThreadPoolExecutor(max_workers=num_workers) as exe:
            futures = [exe.submit(scrape_fn, chunk, progress) for chunk in chunks]
            for fut in as_completed(futures):
                results.extend(fut.result())

    path = OUTPUT_DIR / out_file
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.success(f"Wrote {len(results)} {name.lower()} to {path}")


def main():
    run_stage("Jockey", unique_jockeys, scrape_jockeys, "jockey_details.json", num_workers=25)
    run_stage("Horse", unique_horses, scrape_horses, "horse_details.json", num_workers=25)


if __name__ == "__main__":
    main()
