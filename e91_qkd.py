"""
e91_qkd.py
E91 Quantum Key Distribution protocol simulation.

The E91 protocol (Ekert 1991) uses entangled Bell pairs to generate
a shared secret key between Alice and Bob. Security is guaranteed by
the CHSH inequality: if no eavesdropper, S > 2 (theoretically ~2.828).
An eavesdropper disturbs the state, reducing S toward 2 or below.

Functions:
    generate_e91_key       — simulate E91, return key bits + CHSH value
    detect_eavesdropper    — check CHSH S-value for eavesdropper
    encrypt_with_qkd_key   — XOR encryption
    decrypt_with_qkd_key   — XOR decryption (same as encrypt)
    simulate_eavesdropper  — inject Eve as a depolarizing channel
"""

from __future__ import annotations

import random
import math
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# Alice and Bob each choose from 3 measurement bases.
# Alice: a1=0°, a2=45°, a3=90°   (in units of π/4: 0, 1, 2)
# Bob:   b1=45°, b2=90°, b3=135°  (in units of π/4: 1, 2, 3)
# Matching bases (a2=b1=45°, a3=b2=90°) produce key bits.
# Non-matching combinations test the CHSH inequality.

ALICE_BASES = [0, 1, 2]   # 0°, 45°, 90°
BOB_BASES   = [1, 2, 3]   # 45°, 90°, 135°


def _measure_in_basis(angle_units: int) -> QuantumCircuit:
    """Return a 1-qubit circuit that measures in the given basis.

    The basis is specified in multiples of π/4 (45°).

    Args:
        angle_units: 0 = Z basis (standard), 1 = X basis, 2 = Y basis,
                     3 = rotated X basis (135°).

    Returns:
        A 1-qubit, 1-classical-bit circuit with a measurement.
    """
    qc = QuantumCircuit(1, 1)
    # Rotate to the desired basis before measuring in Z
    angle_rad = angle_units * (math.pi / 4)
    qc.ry(-angle_rad, 0)   # RY rotation to align basis with Z
    qc.measure(0, 0)
    return qc


def _build_e91_pair_circuit(alice_basis: int, bob_basis: int) -> QuantumCircuit:
    """Build a 2-qubit circuit: create Bell pair, measure in given bases.

    Args:
        alice_basis: Alice's basis (0–2).
        bob_basis: Bob's basis (1–3).

    Returns:
        2-qubit circuit ready for simulation.
    """
    qc = QuantumCircuit(2, 2)
    # Bell pair |Phi+>
    qc.h(0)
    qc.cx(0, 1)
    # Alice measures qubit 0 in her basis
    angle_a = alice_basis * (math.pi / 4)
    qc.ry(-angle_a, 0)
    qc.measure(0, 0)
    # Bob measures qubit 1 in his basis
    angle_b = bob_basis * (math.pi / 4)
    qc.ry(-angle_b, 1)
    qc.measure(1, 1)
    return qc


def _run_pair(alice_basis: int, bob_basis: int, noise_model: Optional[NoiseModel] = None) -> tuple[int, int]:
    """Simulate one Bell pair measurement and return (alice_bit, bob_bit).

    Args:
        alice_basis: Alice's measurement basis.
        bob_basis: Bob's measurement basis.
        noise_model: Optional noise model for eavesdropper simulation.

    Returns:
        (alice_bit, bob_bit) each 0 or 1.
    """
    qc = _build_e91_pair_circuit(alice_basis, bob_basis)
    if noise_model:
        sim = AerSimulator(noise_model=noise_model)
    else:
        sim = AerSimulator()
    compiled = transpile(qc, sim)
    job = sim.run(compiled, shots=1)
    counts = job.result().get_counts()
    result = max(counts, key=counts.get)
    # result is "ab" where a=qubit1 (Bob), b=qubit0 (Alice) — LSB order
    bob_bit = int(result[0])    # leftmost = higher index = Bob
    alice_bit = int(result[1])  # rightmost = lower index = Alice
    return alice_bit, bob_bit


def generate_e91_key(
    num_bits: int = 32,
    eavesdropper: bool = False,
) -> dict:
    """Simulate the E91 protocol and generate a shared secret key.

    For each trial:
      - Alice and Bob independently choose random bases
      - They measure their half of a Bell pair
      - If their bases match (a2/b1 or a3/b2), the bit goes into the key
      - Mismatching basis pairs are used to compute the CHSH S-value

    Args:
        num_bits: Desired number of key bits. More trials are run to
                  account for basis mismatches (~1/3 match rate).
        eavesdropper: If True, inject a depolarizing noise channel to
                      simulate Eve's interference.

    Returns:
        A dict with keys:
            "alice_key" : list[int] — Alice's key bits
            "bob_key"   : list[int] — Bob's key bits (should match Alice's)
            "chsh_s"    : float     — CHSH S-value (>2 = no Eve)
            "num_trials": int       — total pairs generated
            "basis_data": list[dict] — raw trial data for analysis
    """
    noise_model = None
    if eavesdropper:
        # Eve intercepts and re-sends, modelled as depolarizing error
        noise_model = NoiseModel()
        eve_error_1q = depolarizing_error(0.15, 1)
        eve_error_2q = depolarizing_error(0.15, 2)
        noise_model.add_all_qubit_quantum_error(eve_error_1q, ["h", "ry"])
        noise_model.add_all_qubit_quantum_error(eve_error_2q, ["cx"])
    alice_key: list[int] = []
    bob_key: list[int] = []
    basis_data: list[dict] = []

    # Run enough trials to collect num_bits key bits
    # Roughly 1/3 of trials produce matching bases
    max_trials = num_bits * 6
    trials = 0

    while len(alice_key) < num_bits and trials < max_trials:
        a_basis = random.choice(ALICE_BASES)
        b_basis = random.choice(BOB_BASES)

        alice_bit, bob_bit = _run_pair(a_basis, b_basis, noise_model)
        trials += 1

        basis_data.append({
            "alice_basis": a_basis,
            "bob_basis": b_basis,
            "alice_bit": alice_bit,
            "bob_bit": bob_bit,
        })

        # Matching bases: a2(45°)=1 / b1(45°)=1 OR a3(90°)=2 / b2(90°)=2
        if (a_basis == 1 and b_basis == 1) or (a_basis == 2 and b_basis == 2):
            alice_key.append(alice_bit)
            # In |Phi+> state, matching-basis measurements are ANTI-correlated
            # for some conventions; we use bob_bit directly since our Bell state
            # gives correlated results in aligned bases with our rotation scheme
            bob_key.append(bob_bit)

    chsh_s = _compute_chsh(basis_data)

    return {
        "alice_key": alice_key[:num_bits],
        "bob_key": bob_key[:num_bits],
        "chsh_s": chsh_s,
        "num_trials": trials,
        "basis_data": basis_data,
    }


def _compute_chsh(basis_data: list[dict]) -> float:
    """Compute the CHSH S-value from recorded trial data.

    S = |E(a1,b1) - E(a1,b3)| + |E(a3,b1) + E(a3,b3)|
    where E(a,b) = correlation coefficient = P(same) - P(different)

    For a pure Bell state:  S_max = 2*sqrt(2) ≈ 2.828
    Classical bound:        S <= 2

    Args:
        basis_data: List of trial records with basis/bit fields.

    Returns:
        Estimated CHSH S-value.
    """
    # Collect correlators for the 4 CHSH basis combinations
    # Using Alice bases 0,2 and Bob bases 1,3 (a1b1, a1b3, a3b1, a3b3)
    correlators: dict[tuple[int, int], list[int]] = {
        (0, 1): [], (0, 3): [],
        (2, 1): [], (2, 3): [],
    }

    for trial in basis_data:
        key = (trial["alice_basis"], trial["bob_basis"])
        if key in correlators:
            # +1 if same outcome, -1 if different
            same = 1 if trial["alice_bit"] == trial["bob_bit"] else -1
            correlators[key].append(same)

    def e(a_basis: int, b_basis: int) -> float:
        data = correlators.get((a_basis, b_basis), [])
        if not data:
            return 0.0
        return float(np.mean(data))

    e_a1b1 = e(0, 1)
    e_a1b3 = e(0, 3)
    e_a3b1 = e(2, 1)
    e_a3b3 = e(2, 3)

    s = abs(e_a1b1 - e_a1b3) + abs(e_a3b1 + e_a3b3)
    return s


def detect_eavesdropper(measurements: dict) -> bool:
    """Determine whether an eavesdropper is present based on CHSH S-value.

    Args:
        measurements: Dict returned by `generate_e91_key` (contains "chsh_s").

    Returns:
        True if Eve is detected (S < 2.0), False if channel is secure.
    """
    s_value = measurements.get("chsh_s", 0.0)
    return s_value < 2.0


def encrypt_with_qkd_key(message_bits: list[int], key: list[int]) -> list[int]:
    """XOR-encrypt a list of bits using a QKD key.

    Args:
        message_bits: Plaintext as a list of 0/1 integers.
        key: Key bits (must be at least as long as message_bits).

    Returns:
        Ciphertext as a list of 0/1 integers.
    """
    return [m ^ k for m, k in zip(message_bits, key)]


def decrypt_with_qkd_key(ciphertext: list[int], key: list[int]) -> list[int]:
    """XOR-decrypt a list of bits using a QKD key (same operation as encrypt).

    Args:
        ciphertext: Encrypted bits as a list of 0/1 integers.
        key: Key bits (must match the key used for encryption).

    Returns:
        Plaintext as a list of 0/1 integers.
    """
    return encrypt_with_qkd_key(ciphertext, key)


def simulate_eavesdropper(key_result: dict, intercept_probability: float = 0.5) -> dict:
    """Simulate Eve intercepting the quantum channel.

    Re-runs key generation with an eavesdropper noise model and injects
    the resulting (lower) S-value into a copy of `key_result`.

    Args:
        key_result: Original key result dict from generate_e91_key.
        intercept_probability: How aggressively Eve intercepts (0–1).

    Returns:
        Modified key_result with reduced chsh_s to reflect Eve's presence.
    """
    noisy_result = generate_e91_key(
        num_bits=len(key_result["alice_key"]),
        eavesdropper=True,
    )
    compromised = dict(key_result)
    compromised["chsh_s"] = noisy_result["chsh_s"]
    compromised["eavesdropper"] = True
    return compromised
