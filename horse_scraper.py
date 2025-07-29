"""Grab horse information from fixture list."""

# %%
from __future__ import annotations

import json
import re
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
# ? Extracting Horses Names

unique_horses = results_df["horse_name"].unique()
no_bracket_horses = [re.sub(r"\s*\([^)]*\)$", "", h) for h in unique_horses]


# %%
# ? Opening Page
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

base_url = "https://www.britishhorseracing.com/racing/horses/"
driver.get(base_url)


# %%
# ? Grabbing Horse Details

horse_details = []
main_tab = driver.current_window_handle
wait = WebDriverWait(driver, 15)
i = 0
for horse in unique_horses:
    i += 1
    logger.info(f"Horse {horse}, {i / len(unique_horses)}")
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
