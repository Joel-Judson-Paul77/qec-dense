"""
ibm_backend.py
IBM Quantum Runtime integration for QEC Dense.

Two modes:
  1. FakeBackend  — uses real IBM device noise model, NO internet needed
  2. RealBackend  — runs on actual IBM quantum hardware (needs API token + queue)

Usage:
    python ibm_backend.py --mode fake --message 10
    python ibm_backend.py --mode real --token YOUR_TOKEN --message 10
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from quantum_protocols import full_superdense_circuit, decode_result


# ---------------------------------------------------------------------------
# Mode 1: FakeBackend (no internet, uses real IBM device noise model)
# ---------------------------------------------------------------------------

def run_on_fake_backend(message: str, shots: int = 1024) -> dict:
    """Run superdense coding on a fake IBM backend.

    FakeSherbrooke mimics the real ibm_sherbrooke 127-qubit device noise
    without needing an IBM account or internet connection. Great for demos.

    Args:
        message: 2-bit message to send.
        shots: Number of measurement shots.

    Returns:
        Dict with counts, decoded message, and accuracy.
    """
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit import transpile

    print(f"[IBM Fake] Using FakeSherbrooke (127-qubit device noise model)")
    print(f"[IBM Fake] Sending message '{message}' ...")

    circuit = full_superdense_circuit(message)
    backend = FakeSherbrooke()

    # Transpile for the fake backend's native gate set
    transpiled = transpile(circuit, backend=backend, optimization_level=1)
    print(f"[IBM Fake] Transpiled circuit: {transpiled.count_ops()}")

    sampler = Sampler(backend)
    job = sampler.run([transpiled], shots=shots)
    result = job.result()

    # Extract counts from SamplerV2 result format.
    # The DataBin attribute name matches the classical register name in the circuit.
    pub_result = result[0]
    # Get the first available bit array (works regardless of register name)
    bit_array = next(iter(pub_result.data.values()))
    counts_raw = bit_array.get_counts()

    decoded = decode_result(counts_raw, with_qec=False)
    correct = sum(v for k, v in counts_raw.items()
                  if decode_result({k: v}, with_qec=False) == message)
    accuracy = correct / shots

    print(f"[IBM Fake] Decoded: '{decoded}' | Accuracy: {accuracy*100:.1f}%")
    print(f"[IBM Fake] Counts: {dict(sorted(counts_raw.items()))}")

    return {
        "backend": "FakeSherbrooke",
        "message": message,
        "decoded": decoded,
        "accuracy": accuracy,
        "counts": counts_raw,
    }


# ---------------------------------------------------------------------------
# Mode 2: Real IBM Quantum hardware
# ---------------------------------------------------------------------------

def run_on_real_hardware(
    message: str,
    token: str,
    backend_name: str = "least_busy",
    shots: int = 1024,
) -> dict:
    """Run superdense coding on real IBM quantum hardware.

    Submits a job to IBM Quantum cloud and waits for results.
    Note: there may be a queue wait depending on backend load.

    Args:
        message: 2-bit message to send.
        token: IBM Quantum API token from quantum.ibm.com.
        backend_name: Backend name (e.g. "ibm_brisbane") or "least_busy".
        shots: Number of shots.

    Returns:
        Dict with counts, decoded message, accuracy, and job ID.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit import transpile

    print(f"[IBM Real] Connecting to IBM Quantum ...")
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=token,
    )

    if backend_name == "least_busy":
        # Find least busy backend with >= 5 qubits that supports dynamic circuits
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=5,
        )
        print(f"[IBM Real] Least busy backend: {backend.name}")
    else:
        backend = service.backend(backend_name)
        print(f"[IBM Real] Using backend: {backend.name}")

    circuit = full_superdense_circuit(message)
    transpiled = transpile(circuit, backend=backend, optimization_level=1)
    print(f"[IBM Real] Circuit transpiled. Gates: {transpiled.count_ops()}")
    print(f"[IBM Real] Submitting job (may queue) ...")

    sampler = Sampler(backend)
    job = sampler.run([transpiled], shots=shots)
    print(f"[IBM Real] Job ID: {job.job_id()}")
    print(f"[IBM Real] Waiting for results ...")

    result = job.result()
    pub_result = result[0]
    bit_array = next(iter(pub_result.data.values()))
    counts_raw = bit_array.get_counts()

    decoded = decode_result(counts_raw, with_qec=False)
    correct = sum(v for k, v in counts_raw.items()
                  if decode_result({k: v}, with_qec=False) == message)
    accuracy = correct / shots

    print(f"[IBM Real] Decoded: '{decoded}' | Accuracy: {accuracy*100:.1f}%")
    print(f"[IBM Real] Counts: {dict(sorted(counts_raw.items()))}")

    return {
        "backend": backend.name,
        "job_id": job.job_id(),
        "message": message,
        "decoded": decoded,
        "accuracy": accuracy,
        "counts": counts_raw,
    }


# ---------------------------------------------------------------------------
# Comparison: Fake vs AerSimulator
# ---------------------------------------------------------------------------

def compare_fake_vs_aer(shots: int = 1024) -> None:
    """Run all 4 messages on FakeSherbrooke vs AerSimulator and compare.

    Shows that real device noise (via fake backend) requires QEC more than
    idealized simulation noise.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile as qiskit_transpile

    print("\n" + "=" * 60)
    print("Comparison: AerSimulator (ideal) vs FakeSherbrooke (device noise)")
    print("=" * 60)

    messages = ["00", "01", "10", "11"]
    print(f"\n{'Message':>8} | {'Aer (ideal)':>12} | {'IBM Fake (noisy)':>16}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*16}")

    for msg in messages:
        # Ideal simulation
        circuit = full_superdense_circuit(msg)
        aer_sim = AerSimulator()
        compiled = qiskit_transpile(circuit, aer_sim)
        counts_aer = aer_sim.run(compiled, shots=shots).result().get_counts()
        acc_aer = sum(v for k, v in counts_aer.items()
                      if decode_result({k: v}, with_qec=False) == msg) / shots

        # Fake backend
        result_fake = run_on_fake_backend(msg, shots=shots)
        acc_fake = result_fake["accuracy"]

        print(f"{msg:>8} | {acc_aer*100:>11.1f}% | {acc_fake*100:>15.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QEC Dense — IBM Quantum Runtime backend",
    )
    parser.add_argument("--mode", choices=["fake", "real", "compare"],
                        default="fake",
                        help="Run mode: fake (no internet), real (IBM cloud), compare")
    parser.add_argument("--message", "-m", default="10",
                        choices=["00", "01", "10", "11"])
    parser.add_argument("--token", default=None,
                        help="IBM Quantum API token (required for --mode real)")
    parser.add_argument("--backend", default="least_busy",
                        help="IBM backend name or 'least_busy'")
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()

    if args.mode == "fake":
        run_on_fake_backend(args.message, args.shots)

    elif args.mode == "real":
        if not args.token:
            print("ERROR: --token required for real mode.")
            print("Get your token at: https://quantum.ibm.com")
            sys.exit(1)
        run_on_real_hardware(args.message, args.token, args.backend, args.shots)

    elif args.mode == "compare":
        compare_fake_vs_aer(args.shots)


if __name__ == "__main__":
    main()
