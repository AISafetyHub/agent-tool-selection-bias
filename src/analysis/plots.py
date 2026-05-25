"""Generate PDF figures from aggregated metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")


def plot_hsr_curve(results: dict, k_values: list[int] = None, output_path: Path = None):
    """HSR@k curve across models."""
    k_values = k_values or [1, 3, 5]
    overall = results["overall"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted(overall):
        ys = [overall[model]["hsr"].get(k, 0.0) for k in k_values]
        ax.plot(k_values, ys, marker="o", label=model)

    ax.set_xlabel("k (turns)")
    ax.set_ylabel("HSR@k")
    ax.set_title("Higher-Privilege Switch Rate across Models")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_domain_heatmap(results: dict, k: int = 5, output_path: Path = None):
    """Heatmap: model x domain HSR@k."""
    by_domain = results["by_domain"]
    models = sorted(by_domain)
    domains = sorted({d for m in by_domain.values() for d in m})

    matrix = []
    for model in models:
        row = [by_domain[model].get(d, {}).get("hsr", {}).get(k, 0.0) for d in domains]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5)))
    sns.heatmap(
        matrix, annot=True, fmt=".0%", cmap="YlOrRd",
        xticklabels=domains, yticklabels=models, ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_title(f"HSR@{k} by Model and Domain")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ped_distribution(results: dict, output_path: Path = None):
    """PED distribution histogram across all models."""
    overall = results["overall"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted(overall):
        values = overall[model]["ped"]["values"]
        if values:
            ax.hist(values, bins=range(0, 5), alpha=0.5, label=model, align="left")

    ax.set_xlabel("PED (distinct standard tools before escalation)")
    ax.set_ylabel("Count")
    ax.set_title("Pre-Escalation Exploration Depth Distribution")
    ax.set_xticks(range(0, 4))
    ax.legend(fontsize=7)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results: dict, output_dir: Path, k_values: list[int] = None):
    """Generate all figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = k_values or [1, 3, 5]

    plot_hsr_curve(results, k_values, output_dir / "hsr_curve.pdf")
    plot_domain_heatmap(results, k=max(k_values), output_path=output_dir / "domain_heatmap.pdf")
    plot_ped_distribution(results, output_dir / "ped_distribution.pdf")
