from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = ROOT / "outputs"


def to_csv(
    data: List[Dict[str, Any]],
    name: str,
    subdir: str = "csv",
) -> str:
    path = OUTPUTS_DIR / subdir / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return str(path)


def to_excel(
    data: List[Dict[str, Any]],
    name: str,
    subdir: str = "excel",
) -> str:
    path = OUTPUTS_DIR / subdir / f"{name}.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)
    return str(path)


def to_json(
    data: Any,
    name: str,
    subdir: str = "json",
) -> str:
    path = OUTPUTS_DIR / subdir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(path)


def load_benchmark(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = str(ROOT / "benchmark" / "questions.csv")
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    elif p.suffix in (".xls", ".xlsx"):
        return pd.read_excel(p)
    else:
        raise ValueError(f"Format non supporté : {p.suffix}")


def summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    numeric = df.select_dtypes(include="number").columns
    desc = df[numeric].describe().round(4)
    return desc
