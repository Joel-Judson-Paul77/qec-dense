"""
bob_server.py
Bob (receiver) — TCP server for QEC Dense superdense coding demo.

Usage:
    python bob_server.py
    python bob_server.py --port 5001
    python bob_server.py --once   # exit after receiving one message

Protocol (matches alice_client.py):
    1. Receive 4-byte big-endian length prefix
    2. Receive exactly that many bytes (QPY-serialized circuit)
    3. Deserialize circuit with qiskit.qpy
    4. Run simulation with AerSimulator
    5. Decode the measurement result
    6. Send "DECODED:<message>\n" acknowledgement back to Alice
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import sys
from typing import Optional

from qiskit import transpile
from qiskit.qpy import load as qpy_load
from qiskit_aer import AerSimulator

from quantum_protocols import decode_result


HOST = "0.0.0.0"   # listen on all interfaces
PORT = 5000
SHOTS = 1024


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Read exactly `num_bytes` from the socket, blocking until done.

    Args:
        sock: Connected TCP socket.
        num_bytes: Number of bytes to read.

    Returns:
        Byte string of length `num_bytes`.

    Raises:
        ConnectionError: If the connection closes before all bytes arrive.
    """
    data = bytearray()
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Connection closed before all data received.")
        data.extend(chunk)
    return bytes(data)


def handle_standard(sock: socket.socket) -> str:
    """Handle a standard (unencrypted) superdense coding transmission.

    Args:
        sock: Connected client socket.

    Returns:
        Decoded 2-bit message string.
    """
    # Read 4-byte length prefix
    length_bytes = recv_exact(sock, 4)
    payload_len = struct.unpack(">I", length_bytes)[0]
    print(f"[Bob] Expecting {payload_len} QPY bytes ...")

    # Read QPY payload
    payload = recv_exact(sock, payload_len)
    print(f"[Bob] Received {len(payload)} bytes. Deserializing circuit ...")

    circuits = qpy_load(io.BytesIO(payload))
    circuit = circuits[0]
    print(f"[Bob] Circuit deserialized: {circuit.num_qubits} qubits, {circuit.num_clbits} classical bits.")

    # Simulate
    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)
    job = simulator.run(compiled, shots=SHOTS)
    counts = job.result().get_counts()
    print(f"[Bob] Simulation counts: {counts}")

    decoded = decode_result(counts, with_qec=False)
    print(f"[Bob] Decoded message: '{decoded}'")
    return decoded


def handle_secure(sock: socket.socket) -> str:
    """Handle a QKD-encrypted superdense coding transmission.

    Args:
        sock: Connected client socket (already read the "SECURE" flag).

    Returns:
        Decoded (and decrypted) 2-bit message string.
    """
    from e91_qkd import decrypt_with_qkd_key

    # Read JSON header
    header_len_bytes = recv_exact(sock, 4)
    header_len = struct.unpack(">I", header_len_bytes)[0]
    header_bytes = recv_exact(sock, header_len)
    header = json.loads(header_bytes.decode("utf-8"))

    key_bits = header["key"]
    s_value = header["s_value"]
    print(f"[Bob] Received QKD key header. CHSH S = {s_value:.4f}")

    if s_value < 2.0:
        print("[Bob] WARNING: S < 2 — possible eavesdropper detected!")

    # Read QPY payload
    payload_len_bytes = recv_exact(sock, 4)
    payload_len = struct.unpack(">I", payload_len_bytes)[0]
    payload = recv_exact(sock, payload_len)

    circuits = qpy_load(io.BytesIO(payload))
    circuit = circuits[0]

    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)
    job = simulator.run(compiled, shots=SHOTS)
    counts = job.result().get_counts()

    cipher_str = decode_result(counts, with_qec=False)
    cipher_bits = [int(b) for b in cipher_str]
    plain_bits = decrypt_with_qkd_key(cipher_bits, key_bits)
    decoded = "".join(str(b) for b in plain_bits)

    print(f"[Bob] Decrypted message: '{decoded}'")
    return decoded


def handle_client(conn: socket.socket, addr: tuple) -> None:
    """Dispatch an incoming connection to the appropriate handler.

    Args:
        conn: Accepted client socket.
        addr: (host, port) of the client.
    """
    print(f"\n[Bob] Connection from {addr[0]}:{addr[1]}")
    try:
        # Peek at first 6 bytes to check for SECURE flag
        first_bytes = recv_exact(conn, 6)
        if first_bytes[:6] == b"SECURE":
            print("[Bob] Secure mode detected.")
            decoded = handle_secure(conn)
        else:
            # Put the 6 bytes back by prepending them to a fake stream
            # We only needed the 4-byte length, so re-inject
            length_prefix = first_bytes[:4]
            extra = first_bytes[4:]
            payload_len = struct.unpack(">I", length_prefix)[0]
            print(f"[Bob] Standard mode. Expecting {payload_len} QPY bytes ...")

            remaining = recv_exact(conn, payload_len - len(extra))
            payload = extra + remaining

            circuits = qpy_load(io.BytesIO(payload))
            circuit = circuits[0]
            print(f"[Bob] Circuit: {circuit.num_qubits}q, {circuit.num_clbits}c")

            simulator = AerSimulator()
            compiled = transpile(circuit, simulator)
            job = simulator.run(compiled, shots=SHOTS)
            counts = job.result().get_counts()
            print(f"[Bob] Counts: {counts}")
            decoded = decode_result(counts, with_qec=False)
            print(f"[Bob] Decoded: '{decoded}'")

        ack = f"DECODED:{decoded}\n"
        conn.sendall(ack.encode("utf-8"))

    except Exception as exc:
        print(f"[Bob] ERROR handling client: {exc}")
        try:
            conn.sendall(f"ERROR:{exc}\n".encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()


def run_server(host: str = HOST, port: int = PORT, once: bool = False) -> None:
    """Start the TCP server and wait for Alice's connections.

    Args:
        host: Interface to bind to.
        port: TCP port to listen on.
        once: If True, handle exactly one connection then exit.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(5)
        print(f"[Bob] Server listening on {host}:{port}")
        print("[Bob] Waiting for Alice ... (Ctrl+C to stop)")

        try:
            while True:
                conn, addr = server_sock.accept()
                handle_client(conn, addr)
                if once:
                    print("[Bob] --once flag set, shutting down.")
                    break
        except KeyboardInterrupt:
            print("\n[Bob] Server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="QEC Dense — Bob (receiver) server")
    parser.add_argument("--host", default=HOST, help=f"Bind address (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"TCP port (default: {PORT})")
    parser.add_argument("--once", action="store_true", help="Handle one connection then exit")
    args = parser.parse_args()
    run_server(args.host, args.port, args.once)


if __name__ == "__main__":
    main()
