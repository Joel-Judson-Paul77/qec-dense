"""
visualization.py
Matplotlib-based plots and circuit diagram exports for QEC Dense.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from qiskit import QuantumCircuit


# Ensure results directory exists
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_accuracy_comparison(
    data_no_qec: list[tuple[float, float]],
    data_with_qec: list[tuple[float, float]],
    filename: str = "accuracy_comparison.png",
    title: str = "QEC Accuracy vs Noise Level",
) -> str:
    """Plot accuracy vs noise level for circuits with and without QEC.

    Args:
        data_no_qec: List of (noise_prob, accuracy) without error correction.
        data_with_qec: List of (noise_prob, accuracy) with error correction.
        filename: Output filename (saved to results/ directory).
        title: Plot title.

    Returns:
        Absolute path to the saved PNG file.
    """
    x_no_qec, y_no_qec = zip(*data_no_qec)
    x_with_qec, y_with_qec = zip(*data_with_qec)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        [x * 100 for x in x_no_qec], y_no_qec,
        color="red", marker="o", linewidth=2, markersize=6,
        label="No Error Correction",
    )
    ax.plot(
        [x * 100 for x in x_with_qec], y_with_qec,
        color="green", marker="s", linewidth=2, markersize=6,
        label="3-Qubit Repetition QEC",
    )

    ax.set_xlabel("Noise Probability (%)", fontsize=13)
    ax.set_ylabel("Decoding Accuracy", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-1, 52)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random guess (50%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_histogram_comparison(
    counts_no_qec: dict[str, int],
    counts_with_qec: dict[str, int],
    message: str,
    filename: str = "histogram_comparison.png",
) -> str:
    """Side-by-side histogram of measurement outcomes with/without QEC.

    Args:
        counts_no_qec: Counts dict from noiseless or noisy run without QEC.
        counts_with_qec: Counts dict from equivalent run with QEC.
        message: The original 2-bit message (for labelling).
        filename: Output filename (saved to results/ directory).

    Returns:
        Absolute path to the saved PNG file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _draw_histogram(ax1, counts_no_qec, f"No QEC — msg '{message}'", "salmon")
    _draw_histogram(ax2, counts_with_qec, f"With QEC — msg '{message}'", "mediumseagreen")

    plt.suptitle(
        f"Measurement Outcomes for Message '{message}'",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_histogram(ax: plt.Axes, counts: dict[str, int], title: str, color: str) -> None:
    """Draw a single histogram on an axis."""
    total = sum(counts.values())
    labels = sorted(counts.keys())
    values = [counts[k] / total for k in labels]

    bars = ax.bar(labels, values, color=color, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Measurement Outcome", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)

    for bar, val in zip(bars, values):
        if val > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )


def save_circuit_diagram(circuit: QuantumCircuit, filename: str) -> str:
    """Save a quantum circuit diagram as a PNG image.

    Args:
        circuit: The QuantumCircuit to draw.
        filename: Output filename (saved to results/ directory).

    Returns:
        Absolute path to the saved PNG file.
    """
    fig = circuit.draw("mpl", style="iqp", fold=40)
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_messages_comparison(
    sweep_data: dict[str, dict[str, list[tuple[float, float]]]],
    filename: str = "all_messages_comparison.png",
) -> str:
    """Plot accuracy vs noise for all 4 messages, with and without QEC.

    Args:
        sweep_data: Dict keyed by message string ("00", "01", "10", "11"),
                    each containing {"no_qec": [...], "with_qec": [...]}.
        filename: Output filename.

    Returns:
        Absolute path to the saved PNG file.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    messages = ["00", "01", "10", "11"]
    colors = {"no_qec": "red", "with_qec": "green"}

    for i, msg in enumerate(messages):
        ax = axes_flat[i]
        data = sweep_data.get(msg, {})
        for key, label in [("no_qec", "No QEC"), ("with_qec", "With QEC")]:
            if key in data:
                xs, ys = zip(*data[key])
                ax.plot(
                    [x * 100 for x in xs], ys,
                    color=colors[key], marker="o", linewidth=2, label=label,
                )
        ax.set_title(f"Message = '{msg}'", fontsize=12, fontweight="bold")
        ax.set_xlabel("Noise (%)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Decoding Accuracy vs Noise — All Messages", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
