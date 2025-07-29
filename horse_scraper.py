#!/usr/bin/env python3
"""Grab horse information from fixture list."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
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

# %%
# ? Setting up params
output_dir = Path("output")
results_dir = output_dir / "results"

results_df = pd.read_csv(results_dir / "results_formatted.csv")

# %%
# ? Extracting Horses Names
unique_horses = results_df["horse_name"].unique()
no_bracket_horses = [re.sub(r"\s*\([^)]*\)$", "", h) for h in unique_horses]

# %%
# ————— Static paths —————
BASE_DIR = Path(__file__).resolve().parent

CHROME_PATH = BASE_DIR / "chrome" / "chrome-win64" / "chrome-win64" / "chrome.exe"
CHROMEDRIVER_PATH = (
    BASE_DIR / "chromedriver" / "chromedriver-win64" / "chromedriver-win64" / "chromedriver.exe"
)
base_url = "https://www.britishhorseracing.com/racing/horses/"


def scrape_horses(horses, progress):
    """
    Spin up a Chrome instance, scrape each horse in `horses`, then quit.
    Returns a list of detail dicts and updates the shared progress bar.
    """
    options = Options()
    options.binary_location = str(CHROME_PATH)
    options.add_argument("--start-maximized")
    options.add_argument("--headless")
    # suppress driver logs
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    service = Service(executable_path=str(CHROMEDRIVER_PATH), log_path="NUL")
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 15)
    main_tab = driver.current_window_handle

    details = []
    for horse in horses:
        logger.info(f"Scraping {horse}")
        try:
            # 1) Search the horse
            driver.get(base_url)
            search_input = wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[1]/div[2]/input'))
            )
            search_input.clear()
            search_input.send_keys(horse)
            go_button = wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[2]'))
            )
            go_button.click()

            # 2) Open the first result
            first_link = wait.until(
                ec.element_to_be_clickable((By.XPATH, '//*[@id="horses-list"]/div/a'))
            )
            href = first_link.get_attribute("href")
            driver.execute_script("window.open(arguments[0]);", href)
            driver.switch_to.window(driver.window_handles[-1])

            # 3) Scrape the general and specific info blocks
            general = wait.until(
                ec.presence_of_element_located((By.XPATH, '//*[@id="horse-single-info"]/div[1]'))
            ).text
            specific = driver.find_element(By.XPATH, '//*[@id="horse-single-info"]/div[2]').text

            # 4) Build a dict with the raw text
            data = {"name": horse}
            lines = general.split("\n")
            if len(lines) > 1:
                desc = lines[1]
                type_part, year_part = desc.split("b.", 1)
                data["type"] = type_part.strip()
                data["birth_year"] = year_part.strip()

            for line in specific.splitlines():
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                if key == "associated_content":
                    continue
                data[key] = val.strip()

            details.append(data)

        except Exception as e:
            logger.error(f"Failed {horse}: {e}")
        finally:
            # 5) Close detail tab and switch back
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(main_tab)
            # update global progress
            progress.update(1)

    driver.quit()
    return details


def main():
    # ? Parallel scrape
    num_workers = 10
    chunk_size = (len(unique_horses) + num_workers - 1) // num_workers
    chunks = [unique_horses[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)]

    horse_details = []
    # create a shared progress bar
    with tqdm(total=len(unique_horses), desc="Overall Progress") as progress:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(scrape_horses, chunk, progress) for chunk in chunks]
            for fut in futures:
                horse_details.extend(fut.result())

    # ? Save the data
    with (output_dir / "horse_details.json").open("w", encoding="utf-8") as f:
        json.dump(horse_details, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
