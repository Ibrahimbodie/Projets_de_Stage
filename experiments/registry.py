from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
LOG_FILE = OUTPUTS_DIR / "logs" / "experiment_log.csv"


class ExperimentLog:
    def __init__(self, name: str = "default"):
        self.name = name
        self.timestamp = datetime.now().isoformat()
        self.parameters: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []

    def set_params(self, **kwargs) -> "ExperimentLog":
        self.parameters.update(kwargs)
        return self

    def record(self, **kwargs) -> None:
        self.results.append(kwargs)

    def to_csv(self, path: Optional[str] = None) -> str:
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(OUTPUTS_DIR / "csv" / f"{self.name}_{ts}.csv")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if self.results:
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)

        return str(out)

    def to_excel(self, path: Optional[str] = None) -> str:
        import pandas as pd

        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(OUTPUTS_DIR / "excel" / f"{self.name}_{ts}.xlsx")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if self.results:
            df = pd.DataFrame(self.results)
            df.to_excel(out, index=False)

        return str(out)

    def to_json(self, path: Optional[str] = None) -> str:
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(OUTPUTS_DIR / "json" / f"{self.name}_{ts}.json")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment": self.name,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "results": self.results,
        }

        with open(out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return str(out)

    def save_all(self) -> Dict[str, str]:
        return {
            "csv": self.to_csv(),
            "excel": self.to_excel(),
            "json": self.to_json(),
        }

    def append_to_global_log(self) -> None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        exists = LOG_FILE.exists()

        summary = {
            "experiment": self.name,
            "timestamp": self.timestamp,
            **self.parameters,
        }

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(summary)

    @staticmethod
    def load_global_log() -> List[Dict[str, str]]:
        if not LOG_FILE.exists():
            return []
        with open(LOG_FILE, "r") as f:
            return list(csv.DictReader(f))
