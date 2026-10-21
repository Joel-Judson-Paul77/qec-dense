"""Measure how often Bob recovers Alice's message, with and without the 3-qubit repetition code.

Uses the same channel-only noise model as the Streamlit app (noise on the transmitted
data qubits only), averages over all four 2-bit messages, and saves a chart to docs/.

    python benchmark.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator

from noise_channel import create_channel_noise_model
from quantum_protocols import full_superdense_circuit, full_superdense_with_qec

MESSAGES = ["00", "01", "10", "11"]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SHOTS = 2048
SEED = 7


def delivery_rate(message: str, p: float, use_qec: bool) -> float:
    circuit = full_superdense_with_qec(message) if use_qec else full_superdense_circuit(message)
    if p == 0:
        sim = AerSimulator(seed_simulator=SEED)
    else:
        sim = AerSimulator(noise_model=create_channel_noise_model("bit_flip", p, [0, 1, 2] if use_qec else [0]),
                           seed_simulator=SEED)
    counts = sim.run(transpile(circuit, sim, optimization_level=0), shots=SHOTS).result().get_counts()
    ok = sum(c for k, c in counts.items() if k.split()[-1][:2] == message)
    return ok / SHOTS


def main() -> None:
    rows = []
    for p in NOISE_LEVELS:
        basic = sum(delivery_rate(m, p, False) for m in MESSAGES) / len(MESSAGES)
        qec = sum(delivery_rate(m, p, True) for m in MESSAGES) / len(MESSAGES)
        rows.append((p, basic, qec))
        print(f"bit-flip p={p:.2f}   no QEC {basic:6.1%}   with QEC {qec:6.1%}")

    os.makedirs("docs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color="#ff003c", label="no QEC")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "s-", color="#00ff88", label="3-qubit repetition code")
    ax.set_xlabel("bit-flip probability on the channel")
    ax.set_ylabel("message delivered")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    ax.set_title(f"Superdense coding over a noisy channel ({SHOTS} shots x 4 messages)")
    fig.tight_layout()
    fig.savefig("docs/delivery_vs_noise.png", dpi=150)
    print("saved docs/delivery_vs_noise.png")


if __name__ == "__main__":
    main()
