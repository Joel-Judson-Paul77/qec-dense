"""
main.py
QEC Dense — Local end-to-end demo (no network required).

Runs all 4 superdense coding messages through:
  1. Noiseless simulation (expect 100% accuracy for all messages)
  2. Noisy simulation sweep (bit_flip 0-50%) with and without QEC
  3. Saves all comparison plots to results/

Run:
    python main.py
"""

from __future__ import annotations

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum_protocols import (
    full_superdense_circuit,
    full_superdense_with_qec,
    decode_result,
    run_noiseless,
)
from noise_channel import (
    run_accuracy_sweep,
    run_channel_noise_sweep,
    create_channel_noise_model,
)
from visualization import (
    plot_accuracy_comparison,
    plot_histogram_comparison,
    save_circuit_diagram,
    plot_all_messages_comparison,
    RESULTS_DIR,
)


MESSAGES = ["00", "01", "10", "11"]
SHOTS = 1024


# ---------------------------------------------------------------------------
# Step 1: Noiseless verification
# ---------------------------------------------------------------------------

def test_noiseless() -> bool:
    """Verify all 4 messages decode correctly with no noise.

    Returns:
        True if all messages passed, False if any failed.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Noiseless Simulation (expect 100% accuracy)")
    print("=" * 60)

    all_passed = True
    simulator = AerSimulator()

    for msg in MESSAGES:
        circuit = full_superdense_circuit(msg)
        compiled = transpile(circuit, simulator)
        job = simulator.run(compiled, shots=SHOTS)
        counts = job.result().get_counts()
        decoded = decode_result(counts, with_qec=False)

        correct = sum(v for k, v in counts.items() if decode_result({k: v}, with_qec=False) == msg)
        accuracy = correct / SHOTS

        status = "PASS" if decoded == msg else "FAIL"
        if decoded != msg:
            all_passed = False

        print(f"  Message '{msg}' -> Decoded '{decoded}' | Accuracy: {accuracy*100:.1f}% | {status}")
        print(f"    Counts: {dict(sorted(counts.items()))}")

    return all_passed


def test_noiseless_qec() -> bool:
    """Verify all 4 messages decode correctly with QEC and no noise."""
    print("\n" + "-" * 60)
    print("STEP 1b: Noiseless Simulation WITH QEC (expect 100% accuracy)")
    print("-" * 60)

    all_passed = True
    simulator = AerSimulator()

    for msg in MESSAGES:
        circuit = full_superdense_with_qec(msg)
        compiled = transpile(circuit, simulator)
        job = simulator.run(compiled, shots=SHOTS)
        counts = job.result().get_counts()
        decoded = decode_result(counts, with_qec=True)

        status = "PASS" if decoded == msg else "FAIL"
        if decoded != msg:
            all_passed = False

        print(f"  Message '{msg}' -> Decoded '{decoded}' | {status}")
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        print(f"    Top counts: {top}")

    return all_passed


# ---------------------------------------------------------------------------
# Step 2: Noise sweep with and without QEC
# ---------------------------------------------------------------------------

def run_noise_sweep(noise_type: str = "bit_flip") -> dict:
    """Run accuracy sweep for all messages with channel-noise model.

    Uses channel-noise-only sweep: noise is applied only to the physical
    data qubits (q0 for basic, q0/q1/q2 for QEC). This is the theoretically
    correct comparison — noise represents decoherence during transmission,
    not gate imperfections.

    Args:
        noise_type: One of "bit_flip", "phase_flip", "depolarizing".

    Returns:
        Dict keyed by message, each containing "no_qec" and "with_qec" data.
    """
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Channel Noise Sweep — {noise_type} (0% to 50%)")
    print("  (Noise applied to data qubits only — idealized QEC scenario)")
    print("=" * 60)

    sweep_data: dict[str, dict] = {}

    for msg in MESSAGES:
        print(f"\n  Message '{msg}' ...")
        no_qec_data = run_channel_noise_sweep(msg, noise_type, with_qec=False, shots=SHOTS)
        with_qec_data = run_channel_noise_sweep(msg, noise_type, with_qec=True, shots=SHOTS)
        sweep_data[msg] = {"no_qec": no_qec_data, "with_qec": with_qec_data}

        # Print table for this message
        print(f"    {'Noise':>8} | {'No QEC':>8} | {'With QEC':>9}")
        print(f"    {'-'*8}-+-{'-'*8}-+-{'-'*9}")
        for (p, a_no), (_, a_qec) in zip(no_qec_data, with_qec_data):
            print(f"    {p*100:>7.0f}% | {a_no*100:>7.1f}% | {a_qec*100:>8.1f}%")

    return sweep_data


# ---------------------------------------------------------------------------
# Step 3: Generate and save all plots
# ---------------------------------------------------------------------------

def save_plots(sweep_data: dict, noise_type: str = "bit_flip") -> None:
    """Generate and save all comparison plots.

    Args:
        sweep_data: Output from run_noise_sweep.
        noise_type: Used for plot titles.
    """
    print(f"\n{'=' * 60}")
    print("STEP 3: Saving Plots")
    print("=" * 60)

    # Average across all 4 messages for the main comparison plot
    noise_levels = [p for p, _ in sweep_data["00"]["no_qec"]]
    avg_no_qec = []
    avg_with_qec = []
    for p in noise_levels:
        no_qec_vals = [dict(sweep_data[m]["no_qec"])[p] for m in MESSAGES]
        with_qec_vals = [dict(sweep_data[m]["with_qec"])[p] for m in MESSAGES]
        avg_no_qec.append((p, sum(no_qec_vals) / len(no_qec_vals)))
        avg_with_qec.append((p, sum(with_qec_vals) / len(with_qec_vals)))

    # Main accuracy comparison plot
    path1 = plot_accuracy_comparison(
        avg_no_qec, avg_with_qec,
        filename="accuracy_comparison.png",
        title=f"QEC Accuracy vs {noise_type.replace('_', ' ').title()} Noise\n(Channel noise on data qubits — avg over all messages)",
    )
    print(f"  Saved: {path1}")

    # Per-message comparison
    path2 = plot_all_messages_comparison(
        sweep_data,
        filename="all_messages_comparison.png",
    )
    print(f"  Saved: {path2}")

    # Histogram comparison: message "10" at 20% noise (channel-only noise)
    print("\n  Generating histogram comparison at 20% bit_flip noise ...")
    circuit_no_qec = full_superdense_circuit("10")
    circuit_with_qec = full_superdense_with_qec("10")

    nm_no_qec = create_channel_noise_model("bit_flip", 0.20, [0])
    nm_with_qec = create_channel_noise_model("bit_flip", 0.20, [0, 1, 2])

    sim_no_qec = AerSimulator(noise_model=nm_no_qec)
    compiled_no = transpile(circuit_no_qec, sim_no_qec, optimization_level=0)
    counts_no_qec = sim_no_qec.run(compiled_no, shots=SHOTS).result().get_counts()

    sim_with_qec = AerSimulator(noise_model=nm_with_qec)
    compiled_qec = transpile(circuit_with_qec, sim_with_qec, optimization_level=0)
    counts_with_qec = sim_with_qec.run(compiled_qec, shots=SHOTS).result().get_counts()

    path3 = plot_histogram_comparison(
        counts_no_qec, counts_with_qec, "10",
        filename="histogram_10_20pct_noise.png",
    )
    print(f"  Saved: {path3}")

    # Circuit diagrams
    print("\n  Saving circuit diagrams ...")
    for msg in MESSAGES:
        circ_basic = full_superdense_circuit(msg)
        path_circ = save_circuit_diagram(circ_basic, f"circuit_{msg}_basic.png")
        print(f"  Saved: {path_circ}")

    circ_qec = full_superdense_with_qec("10")
    path_qec = save_circuit_diagram(circ_qec, "circuit_10_qec.png")
    print(f"  Saved: {path_qec}")


# ---------------------------------------------------------------------------
# Step 4: Summary table
# ---------------------------------------------------------------------------

def print_summary(sweep_data: dict) -> None:
    """Print a formatted summary table of accuracy at key noise levels."""
    print(f"\n{'=' * 60}")
    print("SUMMARY: Average Accuracy at Key Noise Levels")
    print("=" * 60)

    key_levels = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    print(f"  {'Noise':>8} | {'No QEC (avg)':>13} | {'With QEC (avg)':>14} | {'Improvement':>11}")
    print(f"  {'-'*8}-+-{'-'*13}-+-{'-'*14}-+-{'-'*11}")

    for p in key_levels:
        no_qec_vals = [dict(sweep_data[m]["no_qec"]).get(p, 0) for m in MESSAGES]
        with_qec_vals = [dict(sweep_data[m]["with_qec"]).get(p, 0) for m in MESSAGES]
        avg_no = sum(no_qec_vals) / len(no_qec_vals)
        avg_qec = sum(with_qec_vals) / len(with_qec_vals)
        improvement = (avg_qec - avg_no) * 100
        print(
            f"  {p*100:>7.0f}% | {avg_no*100:>12.1f}% | {avg_qec*100:>13.1f}% | "
            f"{improvement:>+10.1f}%"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "#" * 60)
    print("# QEC Dense: Noise-Resilient Quantum Superdense Communication")
    print("# QC\u00b2 Quantum Computing Conclave \u2014 SRM University AP")
    print("# Team: K. Penchala Lokesh, K. Penchala Rakesh,")
    print("#        B. Rahul Sidharth, B. Parvesh, K. Amareswar Rao")
    print("#" * 60)
    # Step 1: Noiseless sanity check
    passed = test_noiseless()
    if not passed:
        print("\nERROR: Noiseless test FAILED. Check quantum_protocols.py.")
        sys.exit(1)

    test_noiseless_qec()

    # Step 2: Noise sweep
    sweep_data = run_noise_sweep(noise_type="bit_flip")

    # Step 3: Save plots
    save_plots(sweep_data, noise_type="bit_flip")

    # Step 4: Summary
    print_summary(sweep_data)

    print(f"\n\nAll results saved to: {RESULTS_DIR}")
    print("Run 'streamlit run app.py' to launch the interactive demo.")
    print("\nDone!")


if __name__ == "__main__":
    main()
