"""Config Template."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import yaml


class Config:
    """Configuration loader for the horse racing scraper."""

    def __init__(self, config_file: str | Path):
        """Config for scraping help."""
        self._config_file = Path(config_file)

        with self._config_file.open("r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        # Parse paths
        self.base_dir: Path = Path(self._raw.get("base_dir", Path.cwd())).resolve()
        self.chrome_path: Path = self._ensure_path(self._raw["chrome_path"])
        self.chromedriver_path: Path = self._ensure_path(self._raw["chromedriver_path"])
        self.output_dir: Path = self._ensure_path(self._raw["output_dir"])
        self.results_dir: Path = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Parse dates
        self.start_date: date = self._parse_date(self._raw["dates"]["start"])
        self.end_date: date = self._parse_date(self._raw["dates"]["end"])

        # Scraping settings
        self.num_workers: int = int(self._raw.get("num_workers", 8))

    def _ensure_path(self, p: str | Path) -> Path:
        """Convert str/Path to absolute Path, expanding relative to base_dir if needed."""
        path = Path(p)
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    @staticmethod
    def _parse_date(d: str | date) -> date:
        """Parse YYYY-MM-DD into datetime.date."""
        if isinstance(d, date):
            return d
        return datetime.strptime(d, "%Y-%m-%d").date()
