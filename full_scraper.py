"""Fixture, Horse, Jockey scraper."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

from horse_racing.config import Config

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date
    from pathlib import Path

_logged_fallbacks: set[str] = set()


@dataclass(frozen=True)
class FixtureRequest:
    """Fixture request metadata."""

    fixture_url: str
    fixture_idx: int
    year: int


@dataclass(frozen=True)
class DetailStageSpec:
    """Configuration for a horse/jockey detail scrape stage."""

    stage_name: str
    scrape_fn: Callable[[list[str], Config], tuple[list[dict], list[str]]]
    out_file: str
    missed_file: Path


# ————— WebDriver factory —————
def make_driver(config: Config) -> webdriver.Chrome:
    """Create and configure a Chrome WebDriver instance."""
    options = Options()
    if config.chrome_path.exists():
        options.binary_location = str(config.chrome_path)
    elif "chrome" not in _logged_fallbacks:
        logger.info(
            "Chrome binary not found at {}. Falling back to Selenium's default Chrome discovery.",
            config.chrome_path,
        )
        _logged_fallbacks.add("chrome")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    if config.chromedriver_path.exists():
        service = Service(executable_path=str(config.chromedriver_path), log_path="NUL")
    elif "chromedriver" not in _logged_fallbacks:
        logger.info(
            "Chromedriver not found at {}. Falling back to Selenium Manager.",
            config.chromedriver_path,
        )
        _logged_fallbacks.add("chromedriver")
        service = Service()
    else:
        service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(20)
    return driver


# ————— Utility —————
def safe_text(
    driver_or_elem: webdriver.Chrome | webdriver.remote.webelement.WebElement,
    xpath: str,
) -> str:
    """Safely retrieve the text content of the first element matching the given XPath."""
    try:
        elem = driver_or_elem.find_element(By.XPATH, xpath)
        return elem.text.strip()
    except NoSuchElementException:
        return "nan"


def parse_fixture_date(date_text: str, year: int) -> date | None:
    """Parse fixture header date text into a concrete date."""
    parsed = pd.to_datetime(f"{date_text} {year}", dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def parse_listing_date(item_text: str, year: int) -> date | None:
    """Best-effort parse of a fixture date from the monthly results list item text."""
    lines = [line.strip() for line in item_text.splitlines() if line.strip()]
    header_candidates = lines[:3] if lines else [item_text.strip()]
    patterns = [
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}\s+[A-Za-z]{3,9}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2}\b",
    ]

    for candidate in header_candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if not match:
                continue
            date_text = re.sub(
                r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+",
                "",
                match.group(0),
            )
            parsed = pd.to_datetime(f"{date_text} {year}", dayfirst=True, errors="coerce")
            if not pd.isna(parsed):
                return parsed.date()

    return None


def write_json(path: Path, payload: object) -> None:
    """Write JSON payload with stable formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def expand_fixture_non_runners(driver: webdriver.Chrome) -> bool:
    """Expand the fixture non-runners panel when present."""
    try:
        toggle = driver.find_element(By.CSS_SELECTOR, "a.runners-toggle")
        toggle_classes = toggle.get_attribute("class") or ""
        if "state--triggered" in toggle_classes:
            return True

        driver.execute_script("arguments[0].scrollIntoView(true);", toggle)
        time.sleep(0.1)
        toggle.click()
        WebDriverWait(driver, 2).until(
            lambda current_driver: bool(
                current_driver.find_elements(
                    By.CSS_SELECTOR,
                    "#fixture-nonrunners-list li",
                ),
            )
            or "hidden"
            not in (
                current_driver.find_element(
                    By.ID,
                    "fixture-nonrunners-list",
                ).get_attribute("class")
                or ""
            ),
        )
    except NoSuchElementException:
        return False
    except TimeoutException:
        logger.warning("Timed out waiting for fixture non-runner details to expand.")
        return False
    return True


def extract_fixture_non_runners(driver: webdriver.Chrome) -> list[dict]:
    """Extract fixture-level non-runner summary entries."""
    if not expand_fixture_non_runners(driver):
        return []

    non_runner_list = driver.find_element(By.XPATH, '//*[@id="fixture-nonrunners-list"]')

    race_entries = non_runner_list.find_elements(By.XPATH, "./li")
    non_runners: list[dict] = []
    for race_entry in race_entries:
        race_time = safe_text(race_entry, './/div[contains(@class, "nr-race-time")]')
        entry_nodes = race_entry.find_elements(
            By.XPATH,
            './/div[contains(@class, "non-runner-entry")]',
        )
        for entry_node in entry_nodes:
            horse_name = safe_text(entry_node, './/span[contains(@class, "entry-title")]')
            text_spans = entry_node.find_elements(
                By.XPATH,
                './/span[contains(@class, "entry-text")]/span',
            )
            declared_at = "nan"
            reason = "nan"
            for text_span in text_spans:
                span_text = text_span.text.strip()
                if not span_text:
                    continue
                if span_text.startswith("Reason:"):
                    reason = span_text.replace("Reason:", "", 1).strip() or "nan"
                elif declared_at == "nan":
                    declared_at = span_text
            if horse_name == "nan":
                continue
            non_runners.append(
                {
                    "race_time": race_time,
                    "horse_name": horse_name,
                    "declared_at": declared_at,
                    "reason": reason,
                },
            )
    return non_runners


def log_stage_summary(stage: str, completed: int, missed: int) -> None:
    """Log a clear summary for a pipeline stage."""
    logger.info("{} stage completed {} records.", stage, completed)
    if missed:
        logger.warning("{} stage missed {} records. See output/missed/.", stage, missed)
    else:
        logger.info("{} stage missed 0 records.", stage)


def click_expand_button(
    driver: webdriver.Chrome,
    button: webdriver.remote.webelement.WebElement,
) -> None:
    """Try to expand a race card row and log failures explicitly."""
    driver.execute_script("arguments[0].scrollIntoView(true);", button)
    time.sleep(0.1)
    try:
        button.click()
    except WebDriverException as exc:
        logger.debug("Could not expand fixture row: {}", exc)
    time.sleep(0.1)


def normalize_cell_text(text: str) -> str:
    """Normalize visible table text while preserving line-level structure."""
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(cleaned_lines) if cleaned_lines else "nan"


def normalize_inline_text(parts: list[str]) -> str:
    """Join visible inline text fragments into a stable single-line value."""
    cleaned_parts = [part.strip() for part in parts if part and part.strip()]
    return " ".join(cleaned_parts) if cleaned_parts else "nan"


def click_elements(
    driver: webdriver.Chrome,
    elements: list[webdriver.remote.webelement.WebElement],
) -> None:
    """Click a list of elements with the same defensive behaviour as other expanders."""
    for element in elements:
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(0.1)
        try:
            element.click()
        except WebDriverException as exc:
            logger.debug("Could not click expandable element: {}", exc)
        time.sleep(0.1)


def extract_race_details(card: webdriver.remote.webelement.WebElement) -> dict[str, str]:
    """Extract flexible race metadata from the local race details list."""
    detail_lists = card.find_elements(By.CSS_SELECTOR, "ul.info-list")
    if not detail_lists:
        return {}

    details: dict[str, str] = {}
    for entry in detail_lists[0].find_elements(By.CSS_SELECTOR, "li.meta-entry"):
        label = normalize_inline_text(
            [
                node.text
                for node in entry.find_elements(By.CSS_SELECTOR, ".entry-title")
            ],
        ).rstrip(":")
        value_nodes = entry.find_elements(
            By.CSS_SELECTOR,
            ".entry-body, .inline-field-value",
        )
        value = normalize_inline_text([node.text for node in value_nodes])
        if label == "nan" or value == "nan":
            continue
        details[label] = value
    return details


def extract_runner_links(
    entry: webdriver.remote.webelement.WebElement,
) -> dict[str, str]:
    """Extract horse, jockey, and trainer links from a runner row."""
    link_map = {
        "horse_url": './/a[contains(@class, "table-link--horse")]',
        "jockey_url": './/a[contains(@class, "table-link--jockey")]',
        "trainer_url": './/a[contains(@class, "table-link--trainer")]',
    }
    links: dict[str, str] = {}
    for key, xpath in link_map.items():
        try:
            href = entry.find_element(By.XPATH, xpath).get_attribute("href")
        except NoSuchElementException:
            href = ""
        links[key] = href or "nan"
    return links


def extract_jockey_name(entry: webdriver.remote.webelement.WebElement) -> str:
    """Extract a jockey name whether or not it is rendered as a clickable link."""
    linked_name = safe_text(entry, './/a[contains(@class, "table-link--jockey")]')
    if linked_name != "nan":
        return linked_name
    return safe_text(entry, './/span[contains(@class, "jockeyname")]')


def extract_distance_time_cell(
    entry: webdriver.remote.webelement.WebElement,
) -> tuple[str, str, str]:
    """Extract the raw distance/timing cell plus its margin and finish-time parts."""
    try:
        cell = entry.find_element(By.CSS_SELECTOR, "div.table-cell.w20.distance")
    except NoSuchElementException:
        return "nan", "nan", "nan"

    distance_nodes = cell.find_elements(
        By.XPATH,
        './/span[contains(@class, "inline-field-value")]//span[not(self::small)]',
    )
    distance_text = normalize_inline_text([node.text for node in distance_nodes])
    finish_time = safe_text(cell, './/small[@title="Finish Time"]')

    raw_parts: list[str] = []
    if distance_text != "nan":
        raw_parts.append(distance_text)
    if finish_time != "nan":
        raw_parts.append(finish_time)
    raw_value = normalize_inline_text(raw_parts)
    return raw_value, distance_text, finish_time


def extract_runner_row(
    entry: webdriver.remote.webelement.WebElement,
    runner_index: int,
) -> tuple[dict, bool]:
    """Extract a runner row while preserving legacy fields and richer table data."""
    cells = entry.find_elements(By.CSS_SELECTOR, "div.table-cell")
    raw_columns = {
        "position": normalize_cell_text(cells[0].text) if len(cells) > 0 else "nan",
        "horse_jockey": normalize_cell_text(cells[1].text) if len(cells) > 1 else "nan",
        "trainer_owner": normalize_cell_text(cells[2].text) if len(cells) > 2 else "nan",
        "distance_time": "nan",
        "sp": normalize_cell_text(cells[4].text) if len(cells) > 4 else "nan",
    }

    raw_distance_time, distance_text, finish_time = extract_distance_time_cell(entry)
    raw_columns["distance_time"] = raw_distance_time

    links = extract_runner_links(entry)
    position_xpath = './/div[contains(@class, "no-draw")]//span[contains(@class, "h1")]'
    owner_xpath = './/div[contains(@class, "trainer")]//small[contains(@class, "ownerName")]'
    table_data = {
        "position_text": safe_text(entry, position_xpath),
        "cloth_number": safe_text(entry, './/small[contains(@class, "cloth-indicator")]'),
        "horse_name": safe_text(entry, './/a[contains(@class, "table-link--horse")]'),
        "horse_url": links["horse_url"],
        "jockey_name": extract_jockey_name(entry),
        "jockey_url": links["jockey_url"],
        "handicap_mark_text": normalize_inline_text(
            [
                node.text
                for node in entry.find_elements(
                    By.XPATH,
                    './/div[contains(@class, "name")]//small[contains(@class, "ownerName")]',
                )
            ],
        ),
        "trainer_name": safe_text(entry, './/a[contains(@class, "table-link--trainer")]'),
        "trainer_url": links["trainer_url"],
        "owner_name": safe_text(entry, owner_xpath),
        "distance_text": distance_text,
        "finish_time": finish_time,
        "sp_text": raw_columns["sp"],
    }

    missing_sections = any(
        value == "nan" for value in (raw_columns["distance_time"], raw_columns["sp"])
    )
    runner_data = {
        "runner_index": runner_index,
        "position": raw_columns["position"],
        "horse_jockey": raw_columns["horse_jockey"],
        "trainer_owner": raw_columns["trainer_owner"],
        "distance_time": raw_columns["distance_time"],
        "sp": raw_columns["sp"],
        "raw_columns": raw_columns,
        "table_data": table_data,
    }
    return runner_data, missing_sections


def extract_races(driver: webdriver.Chrome, fixture_url: str) -> list[dict]:
    """Expand and extract populated race cards from the current fixture page."""
    for btn in driver.find_elements(
        By.XPATH,
        "//*[@id='racecard-ui']/li/div[1]/div[5]/span",
    ):
        click_expand_button(driver, btn)

    raw_cards = driver.find_elements(By.CSS_SELECTOR, "#racecard-ui > li")
    cards = [
        card
        for card in raw_cards
        if card.find_elements(
            By.CSS_SELECTOR,
            "div.table-cell.w10.time span.inline-field-value",
        )
    ]
    races: list[dict] = []
    empty_detail_races: list[str] = []
    runner_section_issues: list[str] = []

    for i, card in enumerate(cards, start=1):
        race_time = card.find_element(By.CSS_SELECTOR, ".table-cell.w10.time").text
        race_name = card.find_element(By.CSS_SELECTOR, ".table-cell.w40.name").text
        race_dist = card.find_element(By.CSS_SELECTOR, ".table-cell.w20.race-distance").text
        winner = card.find_element(By.CSS_SELECTOR, ".table-cell.w25.last-col.winner").text
        race_details = extract_race_details(card)
        if not race_details:
            empty_detail_races.append(f"{i}:{race_time}")

        runners: list[dict] = []
        for j, entry in enumerate(card.find_elements(By.XPATH, "./div[2]//ul/li"), start=1):
            runner_data, missing_sections = extract_runner_row(entry, j)
            if missing_sections:
                runner_section_issues.append(f"{i}:{race_time}:runner_{j}")
            runners.append(runner_data)

        races.append(
            {
                "race_index": i,
                "time": race_time,
                "name": race_name,
                "distance": race_dist,
                "winner_info": winner,
                "race_details": race_details,
                "runners": runners,
            },
        )

    if empty_detail_races:
        logger.warning(
            "Fixture {} had races with empty race details: {}",
            fixture_url,
            ", ".join(empty_detail_races),
        )
    if runner_section_issues:
        logger.warning(
            "Fixture {} had runner rows missing distance/sp sections: {}",
            fixture_url,
            ", ".join(runner_section_issues),
        )
    return races


def extract_horse_rating_histories(driver: webdriver.Chrome) -> dict[str, dict[str, object]]:
    """Extract current ratings and expanded rating histories for a horse."""
    toggle_buttons = driver.find_elements(
        By.CSS_SELECTOR,
        ".expand-handicapperRatingsHistory",
    )
    click_elements(driver, toggle_buttons)

    rating_lists = driver.find_elements(
        By.XPATH,
        '//*[@id="horse-single-info"]/div[2]//ul[contains(@class, "info-list")]',
    )
    ratings: dict[str, dict[str, object]] = {}
    if len(rating_lists) < 2:
        return ratings

    valid_race_types = {"flat", "chase", "hurdle", "awt"}
    for entry in rating_lists[1].find_elements(By.XPATH, "./li"):
        race_type = safe_text(entry, './/span[contains(@class, "entry-title")]').rstrip(":").lower()
        if race_type not in valid_race_types:
            continue
        current_value = safe_text(entry, './/span[contains(@class, "entry-body")]')
        last_published = safe_text(entry, ".//small")
        history: list[dict[str, str]] = []
        history_nodes = entry.find_elements(
            By.CSS_SELECTOR,
            ".handicapperRatingsHistory span",
        )
        for history_node in history_nodes:
            history_text = normalize_inline_text([history_node.text]).replace("\xa0", " ")
            if ":" not in history_text:
                continue
            published_at, rating_value = history_text.split(":", 1)
            history.append(
                {
                    "published_at": published_at.strip(),
                    "rating": rating_value.strip(),
                },
            )
        ratings[race_type] = {
            "current": current_value,
            "last_published": last_published,
            "history": history,
        }
    return ratings


def extract_horse_training_history(driver: webdriver.Chrome) -> list[dict[str, str]]:
    """Extract training history rows from the horse profile table."""
    history_rows = driver.find_elements(
        By.CSS_SELECTOR,
        "#horse-entries-table li.table-entry",
    )
    training_history: list[dict[str, str]] = []
    training_type_xpath = (
        './/span[contains(@class, "inline-field-title") and contains(text(), "Training Type")]'
        "/following-sibling::span"
    )
    trainer_xpath = (
        './/span[contains(@class, "inline-field-title") and contains(text(), "Trainer")]'
        "/following-sibling::span"
    )
    start_date_xpath = (
        './/span[contains(@class, "inline-field-title") and contains(text(), "Start Date")]'
        "/following-sibling::span"
    )
    end_date_xpath = (
        './/span[contains(@class, "inline-field-title") and contains(text(), "End Date")]'
        "/following-sibling::span"
    )
    for row in history_rows:
        trainer_link = row.find_elements(By.XPATH, './/a[contains(@href, "/trainer/#!/")]')
        trainer_url = trainer_link[0].get_attribute("href") if trainer_link else "nan"
        training_history.append(
            {
                "training_type": safe_text(row, training_type_xpath),
                "trainer": safe_text(row, trainer_xpath),
                "trainer_url": trainer_url or "nan",
                "start_date": safe_text(row, start_date_xpath),
                "end_date": safe_text(row, end_date_xpath),
            },
        )
    return training_history


# ————— Scraping helpers —————
def collect_fixture_links(driver: webdriver.Chrome, year: int) -> list[str]:
    """Scroll results list to collect all fixture URLs for the month page."""
    seen_hrefs: set[str] = set()
    logged_sample_row = False
    while True:
        results_list = WebDriverWait(driver, 10).until(
            ec.presence_of_element_located((By.ID, "results-list")),
        )
        WebDriverWait(driver, 10).until(
            lambda current_driver: bool(
                current_driver.find_elements(By.CSS_SELECTOR, "#results-list li"),
            ),
        )
        all_items = results_list.find_elements(By.TAG_NAME, "li") or []
        linked_items = []
        for li in all_items:
            links = li.find_elements(By.TAG_NAME, "a")
            if not links:
                continue
            if not logged_sample_row:
                item_date = parse_listing_date(li.text, year)
                sample_lines = [line.strip() for line in li.text.splitlines() if line.strip()][:6]
                logger.info(
                    "Sample monthly row lines: {} | parsed_date={}",
                    sample_lines,
                    item_date,
                )
                logged_sample_row = True
            linked_items.append(li)
        if not linked_items:
            logger.warning("Results list had no linked fixture rows.")
            break
        hrefs = [li.find_element(By.TAG_NAME, "a").get_attribute("href") for li in linked_items]

        new_hrefs = [href for href in hrefs if href not in seen_hrefs]
        if not new_hrefs:
            break

        seen_hrefs.update(new_hrefs)
        logger.info(f"Scrolled: total unique fixtures so far = {len(seen_hrefs)}")

        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'end', behavior: 'smooth'});",
            linked_items[-1],
        )
        time.sleep(2)

    return sorted(seen_hrefs)


def scrape_fixture(
    driver: webdriver.Chrome,
    request: FixtureRequest,
    main_tab: str,
    config: Config,
) -> tuple[dict | None, bool]:
    """Scrape a single fixture page into structured data."""
    logger.info("Fixture {}: opening {}", request.fixture_idx, request.fixture_url)
    try:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get(request.fixture_url)

        wait = WebDriverWait(driver, 2)
        wait.until(ec.presence_of_element_located((By.ID, "racecard-ui")))

        racecourse_elem = wait.until(
            ec.presence_of_element_located(
                (By.XPATH, '//*[@id="fixture-header-ui"]/h2/div[1]'),
            ),
        )
        racecourse = racecourse_elem.text
        going_text = safe_text(driver, '//*[@id="fixture-header-ui"]/div/div/div[1]')
        weather_text = safe_text(driver, '//*[@id="fixture-header-ui"]/div/div/div[2]')
        other_text = safe_text(driver, '//*[@id="fixture-header-ui"]/div/div/div[3]')
        non_runners = extract_fixture_non_runners(driver)
        races = extract_races(driver, request.fixture_url)
        if not races:
            logger.warning("No populated race cards for fixture {}, skipping.", request.fixture_url)
            return None, True

        fixture_data = {
            "fixture_index": request.fixture_idx,
            "fixture_url": request.fixture_url,
            "date": safe_text(driver, '//*[@id="fixture-header-ui"]/h2/div[2]/span[1]'),
            "year": request.year,
            "racecourse": racecourse,
            "going": going_text,
            "weather": weather_text,
            "other_text": other_text,
            "non_runners": non_runners,
            "races": races,
        }
        fixture_date = parse_fixture_date(fixture_data["date"], request.year)
        if fixture_date is None:
            logger.warning("Could not parse fixture date for {}, skipping.", request.fixture_url)
            return None, False
        if not (config.start_date <= fixture_date <= config.end_date):
            logger.info(
                "Skipping fixture {} because {} is outside configured range {} to {}.",
                request.fixture_url,
                fixture_date,
                config.start_date,
                config.end_date,
            )
            return None, False
    except (TimeoutException, NoSuchElementException) as exc:
        logger.warning(
            "Skipping fixture {} for now due to missing page data: {}",
            request.fixture_url,
            exc,
        )
        return None, True
    except WebDriverException as exc:
        logger.warning("Skipping fixture {} due to scrape error: {}", request.fixture_url, exc)
        return None, False
    else:
        return fixture_data, False
    finally:
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(main_tab)


def process_month_results(
    driver: webdriver.Chrome,
    config: Config,
    year: int,
    month: int,
) -> tuple[list[dict], list[dict]]:
    """Scrape a single monthly results page and retry transient fixture misses once."""
    ym = f"{year}_{month:02d}"
    logger.info("Processing month: {}", ym)

    url = f"https://www.britishhorseracing.com/racing/results/#!?year={year}&month={month}"
    driver.get(url)
    driver.refresh()
    time.sleep(1)

    fixtures = collect_fixture_links(driver, year)
    all_fixtures_data: list[dict] = []
    retry_fixtures: list[FixtureRequest] = []
    main_tab = driver.current_window_handle

    for fixture_idx, fixture_url in enumerate(fixtures, start=1):
        request = FixtureRequest(fixture_url, fixture_idx, year)
        fixture_data, should_retry = scrape_fixture(driver, request, main_tab, config)
        if fixture_data:
            all_fixtures_data.append(fixture_data)
        elif should_retry:
            retry_fixtures.append(request)

    if retry_fixtures:
        logger.info(
            "Retrying {} fixture pages that were missing data on first pass.",
            len(retry_fixtures),
        )

    unresolved_retry_fixtures: list[dict] = []
    for request in retry_fixtures:
        fixture_data, _ = scrape_fixture(driver, request, main_tab, config)
        if fixture_data:
            all_fixtures_data.append(fixture_data)
        else:
            unresolved_retry_fixtures.append(
                {
                    "month": ym,
                    "fixture_index": request.fixture_idx,
                    "fixture_url": request.fixture_url,
                },
            )

    out_file = config.results_dir / f"{ym}.json"
    write_json(out_file, all_fixtures_data)
    logger.info("Wrote {} fixtures to {}", len(all_fixtures_data), out_file)
    return all_fixtures_data, unresolved_retry_fixtures


def retry_missed_fixtures(
    driver: webdriver.Chrome,
    config: Config,
    missed_fixtures: list[dict],
) -> list[dict]:
    """Retry any fixture pages still missed after the monthly pass."""
    if not missed_fixtures:
        write_json(config.missed_dir / "fixtures.json", [])
        return []

    write_json(config.missed_dir / "fixtures.json", missed_fixtures)
    logger.info(
        "Retrying {} missed fixtures at end of full results scrape.",
        len(missed_fixtures),
    )
    final_missed: list[dict] = []
    for missed in missed_fixtures:
        request = FixtureRequest(
            missed["fixture_url"],
            missed["fixture_index"],
            int(missed["month"].split("_")[0]),
        )
        main_tab = driver.current_window_handle
        fixture_data, _ = scrape_fixture(driver, request, main_tab, config)
        if not fixture_data:
            final_missed.append(missed)

    write_json(config.missed_dir / "fixtures.json", final_missed)
    return final_missed


def scrape_monthly_results(config: Config) -> tuple[list[dict], list[dict]]:
    """Scrape BHA race results month by month and save as JSON."""
    driver = make_driver(config)
    all_results: list[dict] = []
    missed_fixtures: list[dict] = []
    month_start = pd.Timestamp(config.start_date).replace(day=1)
    month_end = pd.Timestamp(config.end_date).replace(day=1)
    periods = pd.date_range(start=month_start, end=month_end, freq="MS")

    for dt in periods:
        month_results, month_missed = process_month_results(driver, config, dt.year, dt.month)
        all_results.extend(month_results)
        missed_fixtures.extend(month_missed)

    final_missed = retry_missed_fixtures(driver, config, missed_fixtures)

    driver.quit()
    log_stage_summary("Fixture", len(all_results), len(final_missed))
    return all_results, final_missed


# ————— Part 2: Scrape horse and jockey details —————
def _scrape_single_horse(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    horse: str,
) -> dict | None:
    """Helper to scrape a single horse safely."""
    base_url = "https://www.britishhorseracing.com/racing/horses/"
    try:
        driver.get(base_url)
        time.sleep(1)
        inp = wait.until(
            ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[1]/div[2]/input')),
        )
        inp.clear()
        inp.send_keys(horse)
        time.sleep(1)
        wait.until(
            ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[2]')),
        ).click()

        link = wait.until(
            ec.element_to_be_clickable((By.XPATH, '//*[@id="horses-list"]/div/a')),
        )
        href = link.get_attribute("href")
        driver.get(href)
        logger.info(f"Scraping horse profile: {horse} ({href})")

        general = wait.until(
            ec.presence_of_element_located((By.XPATH, '//*[@id="horse-single-info"]/div[1]')),
        ).text
        specific = driver.find_element(By.XPATH, '//*[@id="horse-single-info"]/div[2]').text
        ratings_history = extract_horse_rating_histories(driver)
        training_history = extract_horse_training_history(driver)

        data: dict = {
            "name": horse,
            "url": href,
            "as_of_date": pd.Timestamp.now().date().isoformat(),
        }
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
        data["ratings_history"] = ratings_history
        data["training_history"] = training_history
    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        logger.error(f"Failed to scrape horse {horse}: {e}")
        return None
    else:
        return data


def scrape_horses(horses_chunk: list[str], config: Config) -> tuple[list[dict], list[str]]:
    """Scrape detailed information for each horse in the list."""
    driver = make_driver(config)
    wait = WebDriverWait(driver, 3)
    details: list[dict] = []
    missed: list[str] = []

    for horse in horses_chunk:
        result = _scrape_single_horse(driver, wait, horse)
        if result:
            details.append(result)
        else:
            missed.append(horse)

    driver.quit()
    return details, missed


def _scrape_single_jockey(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    jockey: str,
) -> dict | None:
    """Helper to scrape a single jockey safely."""
    base_url = "https://www.britishhorseracing.com/racing/participants/jockeys/"
    try:
        driver.get(base_url)
        time.sleep(1)
        inp = wait.until(
            ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[1]//input')),
        )
        inp.clear()
        inp.send_keys(jockey)
        time.sleep(1)
        wait.until(
            ec.element_to_be_clickable((By.XPATH, '//*[@id="searchform"]/div[2]')),
        ).click()

        wait.until(ec.presence_of_element_located((By.XPATH, '//*[@id="jockeys-list"]/div')))
        driver.find_element(By.XPATH, '//*[@id="jockeys-list"]/div').click()

        href = driver.current_url
        logger.info(f"Scraping jockey profile: {jockey} ({href})")

        info = wait.until(
            ec.presence_of_element_located(
                (By.XPATH, '//*[@id="jockey-single-info"]/div[2]/div[2]'),
            ),
        ).text
        career = driver.find_element(
            By.XPATH,
            '//*[@id="jockey-single-info"]/div[2]/div[3]',
        ).text
    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        logger.error(f"Failed to scrape jockey {jockey}: {e}")
        return None
    else:
        return {
            "name": jockey,
            "url": href,
            "as_of_date": pd.Timestamp.now().date().isoformat(),
            "jockey_info": info,
            "career_info": career,
        }


def scrape_jockeys(jockeys_chunk: list[str], config: Config) -> tuple[list[dict], list[str]]:
    """Scrape detailed information for each jockey in the list."""
    driver = make_driver(config)
    wait = WebDriverWait(driver, 3)
    details: list[dict] = []
    missed: list[str] = []

    for jockey in jockeys_chunk:
        result = _scrape_single_jockey(driver, wait, jockey)
        if result:
            details.append(result)
        else:
            missed.append(jockey)

    driver.quit()
    return details, missed


def run_stage(
    name: str,
    items: list[str],
    scrape_fn: Callable[[list[str], Config], tuple[list[dict], list[str]]],
    out_file: str | None,
    config: Config,
) -> tuple[list[dict], list[str]]:
    """Generic runner to scrape details in parallel and save to JSON."""
    num_workers = config.num_workers
    chunk_size = (len(items) + num_workers - 1) // num_workers
    chunks = [items[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)]

    results: list[dict] = []
    missed: list[str] = []
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(scrape_fn, chunk, config) for chunk in chunks]
        for fut in as_completed(futures):
            chunk_results, chunk_missed = fut.result()
            results.extend(chunk_results)
            missed.extend(chunk_missed)

    if out_file is not None:
        path = config.output_dir / out_file
        write_json(path, results)
        logger.success(f"Wrote {len(results)} {name.lower()} details to {path}")
    return results, sorted(set(missed))


def collect_runner_names(all_results: list[dict]) -> tuple[list[str], list[str]]:
    """Collect unique horse and jockey names from scraped fixture results."""
    horse_list: list[str] = []
    jockey_list: list[str] = []

    for fixture in all_results:
        for race in fixture.get("races", []):
            for runner in race.get("runners", []):
                parts = [
                    line.strip()
                    for line in runner.get("horse_jockey", "").splitlines()
                    if line.strip()
                ]
                if not parts:
                    continue
                horse_list.append(parts[0])
                if len(parts) > 1 and parts[1] != "Non Runner":
                    jockey_list.append(parts[1])

    return sorted(set(horse_list)), sorted(set(jockey_list))


def run_detail_stage(
    spec: DetailStageSpec,
    names: list[str],
    config: Config,
) -> None:
    """Run a detail stage, write missed bins, and retry unresolved names once."""
    if not names:
        logger.warning(
            "No {} names found in scraped results; skipping detail stage.",
            spec.stage_name.lower(),
        )
        write_json(spec.missed_file, [])
        return

    results, missed_items = run_stage(spec.stage_name, names, spec.scrape_fn, spec.out_file, config)
    write_json(spec.missed_file, missed_items)
    if not missed_items:
        log_stage_summary(spec.stage_name, len(results), 0)
        return

    logger.info(
        "Retrying {} missed {}s at end of full scrape.",
        len(missed_items),
        spec.stage_name.lower(),
    )
    retry_results, retry_missed = run_stage(
        spec.stage_name,
        missed_items,
        spec.scrape_fn,
        None,
        config,
    )
    merged = {entry["name"]: entry for entry in results}
    merged.update({entry["name"]: entry for entry in retry_results})
    write_json(config.output_dir / spec.out_file, list(merged.values()))
    write_json(spec.missed_file, retry_missed)
    log_stage_summary(spec.stage_name, len(merged), len(retry_missed))


# ————— Main runner —————
def main() -> None:
    """Execute monthly fixture scraping only."""
    config = Config("config.yml")
    all_results, _ = scrape_monthly_results(config)

    if not all_results:
        logger.warning(
            "No fixtures were scraped for the configured date range.",
        )


if __name__ == "__main__":
    main()
