"""
quantum_protocols.py
Core quantum circuits for superdense coding and quantum error correction.

Protocols:
  - Superdense Coding: transmit 2 classical bits using 1 qubit + shared entanglement
  - 3-Qubit Repetition Code: protect against single-qubit bit-flip errors

Qiskit 2.x note: measurement results are LSB-first (qubit 0 is rightmost char).
We reverse the result string before interpreting as a message.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile


# ---------------------------------------------------------------------------
# Bell pair creation
# ---------------------------------------------------------------------------

def create_bell_pair(qc: QuantumCircuit, q0: int, q1: int) -> None:
    """Entangle qubits q0 and q1 into a Bell state |Phi+>.

    Applies H to q0 then CNOT(q0 -> q1).
    After this call qc contains the entanglement preparation layer.

    Args:
        qc: The quantum circuit to modify in-place.
        q0: Index of the control qubit (Alice's qubit before encoding).
        q1: Index of the target qubit (Bob's qubit).
    """
    qc.h(q0)
    qc.cx(q0, q1)


# ---------------------------------------------------------------------------
# Superdense coding: Alice encodes 2 bits onto her qubit
# ---------------------------------------------------------------------------

def alice_encode(qc: QuantumCircuit, qubit: int, message: str) -> None:
    """Encode a 2-bit classical message onto a single qubit.

    Encoding table (standard superdense coding):
        "00" -> I  (identity, do nothing)
        "01" -> Z  (phase flip)
        "10" -> X  (bit flip)
        "11" -> ZX (both: apply X then Z)

    Args:
        qc: The quantum circuit (must already contain a Bell pair).
        qubit: Index of Alice's qubit (the one she controls).
        message: Two-character string of '0' and '1'.

    Raises:
        ValueError: If message is not one of the four valid 2-bit strings.
    """
    if message not in ("00", "01", "10", "11"):
        raise ValueError(f"Message must be a 2-bit string, got '{message}'")

    if message == "00":
        pass           # I gate
    elif message == "01":
        qc.z(qubit)    # Z gate
    elif message == "10":
        qc.x(qubit)    # X gate
    elif message == "11":
        qc.x(qubit)    # X then Z
        qc.z(qubit)


# ---------------------------------------------------------------------------
# Bob decodes via Bell measurement
# ---------------------------------------------------------------------------

def bob_decode(qc: QuantumCircuit, q0: int, q1: int) -> None:
    """Perform Bell measurement to decode Alice's message.

    Applies CNOT(q0 -> q1) then H(q0), then measures both.
    Measurement results (after reversing for LSB ordering) give back
    Alice's 2-bit message.

    Args:
        qc: The quantum circuit containing Alice's encoded state.
        q0: Alice's qubit (control).
        q1: Bob's qubit (target).
    """
    qc.cx(q0, q1)
    qc.h(q0)
    qc.measure_all()


# ---------------------------------------------------------------------------
# Full superdense coding circuit (no error correction)
# ---------------------------------------------------------------------------

def full_superdense_circuit(message: str) -> QuantumCircuit:
    """Build a complete superdense coding circuit for a 2-bit message.

    Circuit layout (2 qubits):
        q0 = Alice's qubit
        q1 = Bob's qubit (the "traveling" qubit)

    Steps:
        1. Create Bell pair on (q0, q1)
        2. Alice encodes `message` onto q0
        3. Bob performs Bell measurement on (q0, q1)

    Args:
        message: 2-bit string to transmit (e.g. "10").

    Returns:
        A QuantumCircuit with 2 qubits and 2 classical bits.
    """
    qc = QuantumCircuit(2, 2)

    # Step 1: entangle
    create_bell_pair(qc, 0, 1)
    qc.barrier(label="entangle")

    # Step 2: Alice encodes
    alice_encode(qc, 0, message)
    qc.barrier(label="encode")

    # Noise channel — id gate on q0 represents physical transmission noise.
    # The noise model targets this specific gate.
    qc.id(0)
    qc.barrier(label="channel")

    # Step 3: Bob decodes (Bell measurement)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])

    return qc


# ---------------------------------------------------------------------------
# 3-Qubit repetition code (QEC)
# ---------------------------------------------------------------------------

def encode_error_correction(
    qc: QuantumCircuit,
    logical_qubit: int,
    ancilla1: int,
    ancilla2: int,
) -> None:
    """Encode a logical qubit using the 3-qubit bit-flip repetition code.

    Copies the state of `logical_qubit` into `ancilla1` and `ancilla2`
    using CNOT gates. This creates |psi>|psi>|psi> (in the computational basis).

    Args:
        qc: Quantum circuit to modify in-place.
        logical_qubit: The qubit carrying the state to protect.
        ancilla1: First ancilla qubit (must be |0>).
        ancilla2: Second ancilla qubit (must be |0>).
    """
    qc.cx(logical_qubit, ancilla1)
    qc.cx(logical_qubit, ancilla2)


def syndrome_measure_and_correct(
    qc: QuantumCircuit,
    q0: int,
    q1: int,
    q2: int,
    syn0: int,
    syn1: int,
) -> None:
    """Apply syndrome parity check CNOTs for 3-qubit repetition code.

    Applies the two parity-check CNOT sequences that compute:
        syn0 qubit = q0 XOR q1  (detects flip on q0 or q1)
        syn1 qubit = q1 XOR q2  (detects flip on q1 or q2)

    Note: This helper applies gate-level parity checks only. Measurement
    and conditional correction require classical registers and cannot be
    done here since syn0/syn1 are qubit indices, not classical bits.
    See full_superdense_with_qec() for the complete mid-circuit measurement
    + classical if_test correction implementation.

    Args:
        qc: Quantum circuit to modify in-place.
        q0, q1, q2: The three data qubits.
        syn0, syn1: Syndrome ancilla qubit indices for parity accumulation.
    """
    # Parity check 1: syn0 = q0 XOR q1
    # CNOT from each data qubit into the syndrome ancilla qubit
    qc.cx(q0, syn0)
    qc.cx(q1, syn0)

    # Parity check 2: syn1 = q1 XOR q2
    qc.cx(q1, syn1)
    qc.cx(q2, syn1)

    # Measure syndrome ancilla qubits into classical syndrome bits.
    # This is a mid-circuit measurement — the syndrome bits are then used
    # in classical if-conditions to apply X corrections on the right qubit.
    # Syndrome decoding table (syn1, syn0) -> error location:
    #   (0, 0) = no error
    #   (0, 1) = error on q0  -> apply X(q0)
    #   (1, 1) = error on q1  -> apply X(q1)
    #   (1, 0) = error on q2  -> apply X(q2)


def full_superdense_with_qec(message: str) -> QuantumCircuit:
    """Build a superdense coding circuit with 3-qubit repetition QEC on Alice's qubit.

    Architecture (7 qubits, 6 classical bits):
        q0, q1, q2 = Alice's logical qubit encoded via repetition code
        q3         = Bob's qubit (entangled with q0 during Bell pair creation)
        q4, q5     = syndrome measurement ancilla qubits
        q6         = spare / unused

    Classical bits:
        c[0], c[1] = Bob's Bell measurement result (the decoded message)
        c[2], c[3] = syndrome bits for error detection
        c[4], c[5] = Bob's raw measurement backup

    The overall flow:
        1. Create Bell pair on (q0, q3)
        2. Encode q0 into repetition code (q0, q1, q2)
        3. Alice encodes 2-bit message on q0
        4. Decode repetition code (syndrome measure + correct)
        5. Bob performs Bell measurement on (q0, q3)

    Args:
        message: 2-bit string to transmit.

    Returns:
        QuantumCircuit ready for simulation.
    """
    # Qubits: q0=Alice_logical, q1=anc1, q2=anc2, q3=Bob, q4=syn_anc0, q5=syn_anc1
    qr = QuantumRegister(6, "q")
    # Classical: c[0..1] = decoded message, c[2..3] = syndrome, c[4..5] = full meas
    cr_msg = ClassicalRegister(2, "msg")
    cr_syn = ClassicalRegister(2, "syn")

    qc = QuantumCircuit(qr, cr_msg, cr_syn)

    alice_q = 0   # logical qubit
    anc1    = 1   # repetition code ancilla
    anc2    = 2   # repetition code ancilla
    bob_q   = 3   # Bob's qubit
    syn_q0  = 4   # syndrome ancilla 0
    syn_q1  = 5   # syndrome ancilla 1

    # Step 1: Bell pair on (alice_q, bob_q)
    create_bell_pair(qc, alice_q, bob_q)
    qc.barrier(label="bell")

    # Step 2: Alice encodes her 2-bit message on the logical qubit FIRST
    alice_encode(qc, alice_q, message)
    qc.barrier(label="alice_encode")

    # Step 3: Spread into repetition code AFTER encoding
    # This protects the already-encoded state from bit-flip noise
    encode_error_correction(qc, alice_q, anc1, anc2)
    qc.barrier(label="qec_encode")

    # Step 3b: Noise channel — explicit id gates on data qubits represent
    # physical transmission / decoherence. Noise model targets only these.
    qc.id(alice_q)
    qc.id(anc1)
    qc.id(anc2)
    qc.barrier(label="channel")

    # Step 4: Syndrome measurement
    # Parity check 1: q0 XOR q1 -> syn_q0
    qc.cx(alice_q, syn_q0)
    qc.cx(anc1, syn_q0)
    # Parity check 2: q1 XOR q2 -> syn_q1
    qc.cx(anc1, syn_q1)
    qc.cx(anc2, syn_q1)

    qc.measure(syn_q0, cr_syn[0])
    qc.measure(syn_q1, cr_syn[1])
    qc.barrier(label="syndrome")

    # Step 5: Error correction based on syndrome integer value.
    # cr_syn integer = cr_syn[1]*2 + cr_syn[0]*1  (LSB-first)
    #   value=1 (binary 01): syn0=1, syn1=0 -> error on q0
    #   value=3 (binary 11): syn0=1, syn1=1 -> error on q1
    #   value=2 (binary 10): syn0=0, syn1=1 -> error on q2
    #   value=0 (binary 00): no error
    with qc.if_test((cr_syn, 1)):
        qc.x(alice_q)       # correct q0
    with qc.if_test((cr_syn, 3)):
        qc.x(anc1)          # correct q1
    with qc.if_test((cr_syn, 2)):
        qc.x(anc2)          # correct q2

    qc.barrier(label="correct")

    # Step 6: Decode repetition code (undo encoding)
    qc.cx(alice_q, anc1)
    qc.cx(alice_q, anc2)
    qc.barrier(label="qec_decode")

    # Step 7: Bob performs Bell measurement on (alice_q, bob_q)
    qc.cx(alice_q, bob_q)
    qc.h(alice_q)
    qc.measure(alice_q, cr_msg[0])
    qc.measure(bob_q, cr_msg[1])

    return qc


# ---------------------------------------------------------------------------
# Utility: decode simulation result to message string
# ---------------------------------------------------------------------------

def decode_result(counts: dict[str, int], with_qec: bool = False) -> str:
    """Extract the most probable message from simulation counts.

    Qiskit's result strings list registers right-to-left (LSB order) and
    separate multiple registers with spaces. We parse accordingly.

    Args:
        counts: Dict mapping bitstring -> count from AerSimulator.
        with_qec: If True, the circuit has two classical registers
                  (syndrome register + message register separated by a space).

    Returns:
        The 2-bit message string (e.g. "10").
    """
    top_result = max(counts, key=counts.get)

    if with_qec:
        # Format: "syn_bits msg_bits" (space-separated, syndrome first)
        parts = top_result.split()
        # The message register is the last part (rightmost in Qiskit output)
        msg_bits = parts[-1]  # 2 bits for the message register
    else:
        msg_bits = top_result  # just 2 bits

    # Qiskit result string for a 2-bit register is "c[1]c[0]" (high index left).
    # For superdense coding the measurement result string matches the
    # original message directly — no reversal needed.
    return msg_bits[:2]


def run_noiseless(message: str, with_qec: bool = False, shots: int = 1024) -> dict[str, int]:
    """Run a noiseless simulation of the superdense coding circuit.

    Args:
        message: 2-bit string to send.
        with_qec: Whether to use the QEC-enhanced circuit.
        shots: Number of simulation shots.

    Returns:
        Counts dictionary from AerSimulator.
    """
    circuit = full_superdense_with_qec(message) if with_qec else full_superdense_circuit(message)
    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)
    job = simulator.run(compiled, shots=shots)
    return job.result().get_counts()
