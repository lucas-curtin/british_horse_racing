import json
import time
from pathlib import Path

from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# %% Setup directories and driver paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
logger.info(f"Output directory ensured at {OUTPUT_DIR}")

CHROME_PATH = BASE_DIR / "chrome" / "chrome-win64" / "chrome-win64" / "chrome.exe"
CHROMEDRIVER_PATH = (
    BASE_DIR / "chromedriver" / "chromedriver-win64" / "chromedriver-win64" / "chromedriver.exe"
)


def make_driver() -> webdriver.Chrome:
    """
    Create and return a Chrome WebDriver with GUI and large resolution.

    Returns:
        webdriver.Chrome: Configured Chrome WebDriver.
    """
    options = Options()
    options.binary_location = str(CHROME_PATH)
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=2560,1440")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    service = Service(executable_path=str(CHROMEDRIVER_PATH), log_path="NUL")
    logger.info("Initializing Chrome WebDriver")
    return webdriver.Chrome(service=service, options=options)


# %% Launch and navigate to fixtures page
driver = make_driver()
logger.info("Driver launched in non-headless mode with high resolution.")
url = "https://www.britishhorseracing.com/racing/fixtures/upcoming/"
driver.get(url)
logger.info(f"Loaded fixtures page: {url}")

# %% Gather all fixture metadata
driver.implicitly_wait(10)
fixture_elements = driver.find_elements(By.CSS_SELECTOR, "#fixtures-list > li > a")
fixtures = []
for idx, el in enumerate(fixture_elements, start=1):
    href = el.get_attribute("href")
    type_date = el.find_element(By.XPATH, "./div[1]").text
    racecourse = el.find_element(By.XPATH, "./div[2]").text
    first_race = el.find_element(By.XPATH, "./div[3]").text
    fixtures.append(
        {"href": href, "type_date": type_date, "racecourse": racecourse, "first_race": first_race}
    )
    logger.info(
        f"Fixture listing {idx}: Type/Date={type_date}, Racecourse={racecourse}, First Race={first_race}, URL={href}"
    )
logger.info(f"Found {len(fixtures)} upcoming fixtures.")

# %% Process each fixture and collect data
main_tab = driver.current_window_handle
wait = WebDriverWait(driver, 30)
all_fixtures = []

for fidx, meta in enumerate(fixtures, start=1):
    href = meta["href"]
    logger.info(f"Processing fixture {fidx}/{len(fixtures)}: {href}")
    driver.execute_script("window.open(arguments[0]);", href)
    driver.switch_to.window(driver.window_handles[-1])

    # Extract header
    wait.until(EC.presence_of_element_located((By.ID, "fixture-header-ui")))
    going = driver.find_element(By.XPATH, '//*[@id="fixture-header-ui"]/div/div/div/div[1]').text
    weather = driver.find_element(By.XPATH, '//*[@id="fixture-header-ui"]/div/div/div/div[2]').text
    other = driver.find_element(By.XPATH, '//*[@id="fixture-header-ui"]/div/div/div/div[3]').text
    logger.info(f"Fixture {fidx}: Going={going}, Weather={weather}, Other={other}")

    # Expand all races
    wait.until(EC.presence_of_element_located((By.ID, "racecard-ui")))
    expand_buttons = driver.find_elements(By.XPATH, '//*[@id="racecard-ui"]/li/div[1]/div[5]')
    logger.info(f"Found {len(expand_buttons)} races to expand in fixture {fidx}.")
    for ridx, btn in enumerate(expand_buttons, start=1):
        try:
            driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            time.sleep(0.1)
            btn.click()
            logger.info(f"Fixture {fidx}: expanded race {ridx}.")
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Fixture {fidx}, failed to expand race {ridx}: {e}")

    # Extract race and runner details
    fixture_data = {
        "type_date": meta["type_date"],
        "racecourse": meta["racecourse"],
        "first_race": meta["first_race"],
        "going": going,
        "weather": weather,
        "other": other,
        "races": [],
    }

    race_cards = driver.find_elements(By.CSS_SELECTOR, "#racecard-ui > li")
    for i, card in enumerate(race_cards, start=1):
        time_txt = card.find_element(By.XPATH, ".//div[1]/div[1]").text
        name_txt = card.find_element(By.XPATH, ".//div[1]/div[2]").text
        dist_txt = card.find_element(By.XPATH, ".//div[1]/div[3]").text
        cond_txt = card.find_element(By.XPATH, ".//div[1]/div[4]").text
        logger.info(
            f"Fixture {fidx}, Race {i}: Time={time_txt}, Name={name_txt}, Distance={dist_txt}, Conditions={cond_txt}"
        )

        race_info = {
            "time": time_txt,
            "name": name_txt,
            "distance": dist_txt,
            "conditions": cond_txt,
            "runners": [],
        }

        runner_rows = card.find_elements(By.XPATH, ".//div[2]/div[2]/div[2]/ul/li")
        for j, row in enumerate(runner_rows, start=1):
            try:
                runner_info = {
                    "no_draw": row.find_element(By.XPATH, ".//div/div[1]").text,
                    "horse_jockey": row.find_element(By.XPATH, ".//div/div[2]").text,
                    "age": row.find_element(By.XPATH, ".//div/div[3]").text,
                    "form_type": row.find_element(By.XPATH, ".//div/div[4]").text,
                    "bha_rating": row.find_element(By.XPATH, ".//div/div[5]").text,
                    "weight": row.find_element(By.XPATH, ".//div/div[6]").text,
                    "trainer_owner": row.find_element(By.XPATH, ".//div/div[7]").text,
                    "odds": row.find_element(By.XPATH, ".//div/div[8]").text,
                }
                race_info["runners"].append(runner_info)
                logger.info(f"Fixture {fidx}, Race {i}, Runner {j}: {runner_info}")
            except Exception as e:
                logger.error(f"Fixture {fidx}, Race {i}, runner {j} extraction failed: {e}")

        fixture_data["races"].append(race_info)

    all_fixtures.append(fixture_data)
    logger.info(f"Completed data extraction for fixture {fidx}.")

    # Close tab and switch back
    driver.close()
    driver.switch_to.window(main_tab)
    logger.info(f"Closed tab for fixture {fidx}, returned to main list.")

# %% Save results to JSON
output_file = OUTPUT_DIR / "fixtures.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_fixtures, f, ensure_ascii=False, indent=2)
logger.info(f"Saved all fixture data to {output_file}")

# %% Cleanup
driver.quit()
logger.info("Driver quit, scraping complete.")
# %%
