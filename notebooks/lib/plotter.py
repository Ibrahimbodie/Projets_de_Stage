from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ROOT / "outputs" / "figures"


def _save(name: str, subdir: str = "comparisons") -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(path)


def set_style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def histogram(
    data: pd.Series,
    title: str,
    xlabel: str,
    filename: str,
    bins: int = 20,
    color: str = "#1F384B",
) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, bins=bins, color=color, edgecolor="white", ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fréquence")
    return _save(filename, "histograms")


def boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    filename: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y, palette="muted", ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    return _save(filename, "boxplots")


def heatmap(
    data: pd.DataFrame,
    title: str,
    filename: str,
    annot: bool = True,
    cmap: str = "Blues",
) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data, annot=annot, fmt=".3f", cmap=cmap,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    return _save(filename, "heatmaps")


def comparison_curve(
    results: Dict[str, List[float]],
    title: str,
    filename: str,
    xlabel: str = "top_k",
    ylabel: str = "Score",
) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values in results.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", label=label)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(filename, "curves")


def barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    filename: str,
    hue: Optional[str] = None,
) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x=x, y=y, hue=hue, palette="muted", ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    return _save(filename, "comparisons")


def pareto(
    data: pd.Series,
    title: str,
    filename: str,
    top_n: int = 10,
) -> str:
    set_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    counts = data.value_counts().head(top_n)
    ax1.bar(range(len(counts)), counts.values, color="#1F384B", alpha=0.8)
    ax1.set_xlabel("Catégorie d'erreur")
    ax1.set_ylabel("Fréquence", color="#1F384B")

    ax2 = ax1.twinx()
    cumsum = np.cumsum(counts.values) / counts.sum() * 100
    ax2.plot(range(len(counts)), cumsum, "r-", marker="o", linewidth=2)
    ax2.set_ylabel("Pourcentage cumulé (%)", color="red")

    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(counts.index, rotation=45, ha="right")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    return _save(filename, "histograms")
