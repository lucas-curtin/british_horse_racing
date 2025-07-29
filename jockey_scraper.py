#!/usr/bin/env python3
"""Grab jockey information from fixture list."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger
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

results_df = pd.read_csv(results_dir / "results_formatted.csv")

# %%
# ? Extracting Jockey Names
unique_jockeys = results_df["jockey_name"].unique()
unique_jockeys = [j for j in unique_jockeys if j != "Non Runner"]

# %%
# ————— Static paths & driver setup —————
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
wait = WebDriverWait(driver, 15)

base_url = "https://www.britishhorseracing.com/racing/participants/jockeys/"

# %%
# ? Loop through jockeys, search & scrape
jockey_details = []

for jockey in unique_jockeys:
    logger.info(f"Jockey {jockey}")
    # navigate back to main jockeys page
    driver.get(base_url)

    # 1) search the jockey
    search_input = wait.until(
        ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[1]//input'))
    )
    search_input.clear()
    search_input.send_keys(jockey)

    go_button = wait.until(ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[2]')))
    go_button.click()

    # 2) click the first result
    wait.until(ec.presence_of_element_located((By.XPATH, '//*[@id="jockeys-list"]/div')))
    first_result = driver.find_element(By.XPATH, '//*[@id="jockeys-list"]/div')
    first_result.click()

    # 3) scrape the details
    wait.until(ec.presence_of_element_located((By.XPATH, '//*[@id="jockey-single-info"]/div[2]')))
    jockey_info = driver.find_element(By.XPATH, '//*[@id="jockey-single-info"]/div[2]/div[2]').text
    career_info = driver.find_element(By.XPATH, '//*[@id="jockey-single-info"]/div[2]/div[3]').text

    jockey_details.append(
        {
            "name": jockey,
            "jockey_info": jockey_info,
            "career_info": career_info,
        }
    )

# %%
# ? Save the data
driver.quit()
with (output_dir / "jockey_details.json").open("w", encoding="utf-8") as f:
    json.dump(jockey_details, f, ensure_ascii=False, indent=4)
# %%
