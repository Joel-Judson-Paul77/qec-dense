"""
noise_channel.py
Qiskit Aer noise models and noisy simulation utilities.

Noise types supported:
  - bit_flip:     X error applied with probability p
  - phase_flip:   Z error applied with probability p
  - depolarizing: random Pauli error (X, Y, or Z) with probability p
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error

from quantum_protocols import (
    full_superdense_circuit,
    full_superdense_with_qec,
    decode_result,
)


def create_noise_model(noise_type: str, probability: float) -> NoiseModel:
    """Build a Qiskit Aer NoiseModel for the requested noise type.

    The error is applied to every 1-qubit gate and every 2-qubit gate
    in the circuit.

    Args:
        noise_type: One of "bit_flip", "phase_flip", "depolarizing".
        probability: Error probability in [0, 1].

    Returns:
        A configured NoiseModel ready to pass to AerSimulator.

    Raises:
        ValueError: If noise_type is not recognised.
    """
    noise_model = NoiseModel()

    if noise_type == "bit_flip":
        # X error with prob p, identity with prob (1-p)
        error_1q = pauli_error([("X", probability), ("I", 1 - probability)])
        # For 2-qubit gates use tensor product of two independent errors
        error_2q = error_1q.tensor(error_1q)
    elif noise_type == "phase_flip":
        error_1q = pauli_error([("Z", probability), ("I", 1 - probability)])
        error_2q = error_1q.tensor(error_1q)
    elif noise_type == "depolarizing":
        error_1q = depolarizing_error(probability, 1)
        error_2q = depolarizing_error(probability, 2)
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Choose from: bit_flip, phase_flip, depolarizing."
        )

    # Apply 1-qubit error to single-qubit gates only
    noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x", "z", "u1", "u2", "u3"])
    # Apply 2-qubit error to two-qubit gates only
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model


def run_noisy_simulation(
    circuit,
    noise_model: NoiseModel,
    shots: int = 1024,
) -> dict[str, int]:
    """Simulate a quantum circuit under a given noise model.

    Args:
        circuit: A QuantumCircuit with measurements.
        noise_model: The NoiseModel to apply during simulation.
        shots: Number of measurement shots.

    Returns:
        A counts dictionary mapping bitstring -> number of occurrences.
    """
    simulator = AerSimulator(noise_model=noise_model)
    compiled = transpile(circuit, simulator)
    job = simulator.run(compiled, shots=shots)
    return job.result().get_counts()


def run_accuracy_sweep(
    message: str,
    noise_type: str,
    noise_range: list[float] | None = None,
    with_qec: bool = False,
    shots: int = 1024,
) -> list[tuple[float, float]]:
    """Sweep noise probability from 0% to 50% and measure decoding accuracy.

    Args:
        message: 2-bit string to transmit at each noise level.
        noise_type: One of "bit_flip", "phase_flip", "depolarizing".
        noise_range: List of noise probabilities to test. Defaults to
                     [0.00, 0.05, 0.10, ..., 0.50].
        with_qec: Whether to use the QEC-enhanced circuit.
        shots: Number of shots per noise level.

    Returns:
        List of (noise_probability, accuracy_fraction) tuples where
        accuracy_fraction is in [0, 1].
    """
    if noise_range is None:
        noise_range = [round(i * 0.05, 2) for i in range(11)]  # 0.0 to 0.5

    results: list[tuple[float, float]] = []

    circuit = full_superdense_with_qec(message) if with_qec else full_superdense_circuit(message)

    for prob in noise_range:
        if prob == 0.0:
            # Noiseless simulation
            simulator = AerSimulator()
            compiled = transpile(circuit, simulator)
            job = simulator.run(compiled, shots=shots)
            counts = job.result().get_counts()
        else:
            noise_model = create_noise_model(noise_type, prob)
            counts = run_noisy_simulation(circuit, noise_model, shots=shots)

        correct_count = sum(
            v for k, v in counts.items()
            if decode_result({k: v}, with_qec=with_qec) == message
        )
        accuracy = correct_count / shots
        results.append((prob, accuracy))

    return results


def _expected_bitstring(message: str, with_qec: bool) -> str:
    """Return the expected top-count bitstring for a given message.

    Qiskit measures LSB-first so message "10" appears as "01" in the
    counts string. With QEC there is also a syndrome register prepended.

    Args:
        message: Original 2-bit message string.
        with_qec: Whether the circuit has the QEC syndrome register.

    Returns:
        The expected bitstring key in the counts dictionary.
    """
    # The result string matches the message directly (see decode_result).
    if with_qec:
        # Format: "syn_bits msg_bits" — syndrome should be "00" (no error)
        return f"00 {message}"
    else:
        return message


def create_channel_noise_model(noise_type: str, probability: float, data_qubits: list[int]) -> NoiseModel:
    """Build a noise model that only affects specific data qubits.

    This models the realistic QEC scenario where noise represents decoherence
    on the physical qubits being transmitted, while control gates (CNOT for
    Bell pair, encoding, syndrome) remain ideal. This is the appropriate model
    to compare 1-qubit unprotected vs 3-qubit repetition code.

    Args:
        noise_type: One of "bit_flip", "phase_flip", "depolarizing".
        probability: Error probability per physical qubit.
        data_qubits: List of qubit indices to apply noise to.

    Returns:
        A NoiseModel with single-qubit errors on specified qubits only.
    """
    noise_model = NoiseModel()

    if noise_type == "bit_flip":
        error = pauli_error([("X", probability), ("I", 1 - probability)])
    elif noise_type == "phase_flip":
        error = pauli_error([("Z", probability), ("I", 1 - probability)])
    elif noise_type == "depolarizing":
        error = depolarizing_error(probability, 1)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'.")

    # Apply noise only to the 'id' (identity) gates — these are explicitly
    # inserted in the circuit to mark the "transmission channel" phase.
    for q in data_qubits:
        noise_model.add_quantum_error(error, ["id"], [q])

    return noise_model


def run_channel_noise_sweep(
    message: str,
    noise_type: str,
    noise_range: list[float] | None = None,
    with_qec: bool = False,
    shots: int = 1024,
) -> list[tuple[float, float]]:
    """Accuracy sweep applying noise only to the logical data qubit(s).

    For the 'no QEC' case: noise is applied only to qubit 0 (Alice's qubit).
    For the 'with QEC' case: noise is applied to qubits 0, 1, 2 independently
    (the three physical qubits of the repetition code).

    This is the idealized model that demonstrates QEC's theoretical advantage:
    at error rate p per qubit, QEC reduces logical error from p to ~3p^2.

    Args:
        message: 2-bit string to transmit.
        noise_type: Type of noise.
        noise_range: Noise probabilities to test.
        with_qec: Whether to use the QEC-enhanced circuit.
        shots: Shots per noise level.

    Returns:
        List of (noise_prob, accuracy) tuples.
    """
    if noise_range is None:
        noise_range = [round(i * 0.05, 2) for i in range(11)]

    # Data qubits differ between basic and QEC circuits
    # Basic circuit: q0=Alice, q1=Bob — apply noise to q0 (the transmitted qubit)
    # QEC circuit: q0,q1,q2=Alice's logical, q3=Bob — apply noise to q0,q1,q2
    data_qubits_no_qec = [0]       # 1 physical qubit
    data_qubits_with_qec = [0, 1, 2]  # 3 physical qubits

    data_qubits = data_qubits_with_qec if with_qec else data_qubits_no_qec
    circuit = full_superdense_with_qec(message) if with_qec else full_superdense_circuit(message)

    results: list[tuple[float, float]] = []
    simulator = AerSimulator()

    for prob in noise_range:
        if prob == 0.0:
            compiled = transpile(circuit, simulator, optimization_level=0)
            job = simulator.run(compiled, shots=shots)
            counts = job.result().get_counts()
        else:
            nm = create_channel_noise_model(noise_type, prob, data_qubits)
            sim = AerSimulator(noise_model=nm)
            # optimization_level=0 preserves id gates so noise is applied
            compiled = transpile(circuit, sim, optimization_level=0)
            job = sim.run(compiled, shots=shots)
            counts = job.result().get_counts()

        # Count all shots where the message was decoded correctly,
        # regardless of syndrome bits (corrected errors have non-zero syndrome).
        correct_count = sum(
            v for k, v in counts.items()
            if decode_result({k: v}, with_qec=with_qec) == message
        )
        accuracy = correct_count / shots
        results.append((prob, accuracy))

    return results


def compare_noise_sweep(
    noise_type: str = "bit_flip",
    shots: int = 1024,
) -> dict[str, list[tuple[float, float]]]:
    """Run accuracy sweeps for all 4 messages with and without QEC.

    Args:
        noise_type: Type of noise channel to test.
        shots: Shots per noise level per message.

    Returns:
        Dict with keys "no_qec" and "with_qec", each mapping to a list of
        (noise_prob, avg_accuracy_across_4_messages) tuples.
    """
    messages = ["00", "01", "10", "11"]
    noise_range = [round(i * 0.05, 2) for i in range(11)]

    # Accumulate accuracy per noise level across all messages
    acc_no_qec = {p: [] for p in noise_range}
    acc_with_qec = {p: [] for p in noise_range}

    for msg in messages:
        sweep_no_qec = run_channel_noise_sweep(msg, noise_type, noise_range, with_qec=False, shots=shots)
        sweep_with_qec = run_channel_noise_sweep(msg, noise_type, noise_range, with_qec=True, shots=shots)

        for (prob, acc) in sweep_no_qec:
            acc_no_qec[prob].append(acc)
        for (prob, acc) in sweep_with_qec:
            acc_with_qec[prob].append(acc)

    avg_no_qec = [(p, float(np.mean(accs))) for p, accs in sorted(acc_no_qec.items())]
    avg_with_qec = [(p, float(np.mean(accs))) for p, accs in sorted(acc_with_qec.items())]

    return {"no_qec": avg_no_qec, "with_qec": avg_with_qec}
