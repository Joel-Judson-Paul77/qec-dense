"""
alice_client.py
Alice (sender) — TCP client for QEC Dense superdense coding demo.

Usage:
    python alice_client.py --message 10
    python alice_client.py --message 11 --host 192.168.1.5
    python alice_client.py --message 00 --secure   # uses E91 QKD key

Protocol:
    1. Build the superdense coding circuit for `message`
    2. Serialize the circuit using qiskit.qpy
    3. Send a 4-byte big-endian length prefix followed by the QPY bytes
    4. Wait for Bob's acknowledgement (decoded message string)
    5. Print confirmation
"""

from __future__ import annotations

import argparse
import io
import socket
import struct
import sys

from qiskit.qpy import dump as qpy_dump

from quantum_protocols import full_superdense_circuit


HOST = "127.0.0.1"
PORT = 5000
BUFFER_SIZE = 65536


def serialize_circuit(message: str) -> bytes:
    """Build the superdense coding circuit and serialize it via QPY.

    Args:
        message: 2-bit string to encode (e.g. "10").

    Returns:
        Raw QPY bytes representing the circuit.
    """
    circuit = full_superdense_circuit(message)
    buf = io.BytesIO()
    qpy_dump(circuit, buf)
    return buf.getvalue()


def send_message(message: str, host: str = HOST, port: int = PORT) -> str:
    """Encode a 2-bit message and send it to Bob via TCP.

    The wire format is:
        [4 bytes: big-endian uint32 payload length][N bytes: QPY payload]

    Args:
        message: 2-bit string to send ("00", "01", "10", or "11").
        host: Bob's IP address or hostname.
        port: TCP port Bob is listening on.

    Returns:
        Bob's decoded message string (should match `message` if no errors).

    Raises:
        ConnectionRefusedError: If Bob's server is not running.
    """
    payload = serialize_circuit(message)
    length_prefix = struct.pack(">I", len(payload))

    print(f"[Alice] Connecting to Bob at {host}:{port} ...")
    with socket.create_connection((host, port), timeout=10) as sock:
        print(f"[Alice] Connected. Sending message '{message}' ({len(payload)} QPY bytes).")
        sock.sendall(length_prefix + payload)

        # Wait for Bob's ack: a short UTF-8 string like "DECODED:10\n"
        ack = sock.recv(256).decode("utf-8").strip()
        print(f"[Alice] Bob acknowledged: {ack}")

    if ack.startswith("DECODED:"):
        return ack.split(":", 1)[1].strip()
    return ack


def send_secure_message(message: str, host: str = HOST, port: int = PORT) -> str:
    """Send a message encrypted with an E91 QKD key.

    Requires e91_qkd.py. The encrypted payload is XOR-encrypted before
    building the superdense circuit, and the QKD key metadata is sent
    as a JSON header before the QPY payload.

    Args:
        message: 2-bit string to send.
        host: Bob's host.
        port: Bob's port.

    Returns:
        Bob's decoded message.
    """
    from e91_qkd import generate_e91_key, encrypt_with_qkd_key
    import json

    print("[Alice] Generating E91 QKD key ...")
    key_result = generate_e91_key(num_bits=8)
    key_bits = key_result["alice_key"]
    s_value = key_result["chsh_s"]
    print(f"[Alice] CHSH S = {s_value:.4f} ({'SECURE' if s_value > 2 else 'WARNING: Eve detected!'})")

    # XOR-encrypt the 2-bit message using the first 2 bits of the key
    msg_bits = [int(b) for b in message]
    cipher_bits = encrypt_with_qkd_key(msg_bits, key_bits[:2])
    cipher_str = "".join(str(b) for b in cipher_bits)

    payload = serialize_circuit(cipher_str)
    header = json.dumps({"mode": "secure", "key": key_bits[:2], "s_value": s_value}).encode()
    header_len = struct.pack(">I", len(header))
    payload_len = struct.pack(">I", len(payload))

    print(f"[Alice] Sending encrypted message '{cipher_str}' to {host}:{port} ...")
    with socket.create_connection((host, port), timeout=10) as sock:
        # Protocol: SECURE flag + header length + header + payload length + payload
        sock.sendall(b"SECURE")
        sock.sendall(header_len + header)
        sock.sendall(payload_len + payload)

        ack = sock.recv(256).decode("utf-8").strip()
        print(f"[Alice] Bob acknowledged: {ack}")

    return ack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QEC Dense — Alice (sender) client",
    )
    parser.add_argument(
        "--message", "-m",
        default="10",
        choices=["00", "01", "10", "11"],
        help="2-bit message to send (default: 10)",
    )
    parser.add_argument(
        "--host",
        default=HOST,
        help=f"Bob's IP address (default: {HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"TCP port (default: {PORT})",
    )
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Use E91 QKD key for encryption",
    )
    args = parser.parse_args()

    try:
        if args.secure:
            decoded = send_secure_message(args.message, args.host, args.port)
        else:
            decoded = send_message(args.message, args.host, args.port)

        if decoded == args.message:
            print(f"[Alice] SUCCESS: Bob decoded '{decoded}' correctly!")
            sys.exit(0)
        else:
            print(f"[Alice] MISMATCH: sent '{args.message}', Bob decoded '{decoded}'")
            sys.exit(1)

    except ConnectionRefusedError:
        print(f"[Alice] ERROR: Could not connect to {args.host}:{args.port}.")
        print("  Make sure Bob's server is running first:  python bob_server.py")
        sys.exit(2)
    except Exception as exc:
        print(f"[Alice] ERROR: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
