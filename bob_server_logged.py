"""
bob_server_logged.py
Wrapper around bob_server.py that writes all output to a JSON log file
so the Streamlit UI can read results without blocking on subprocess stdout.

Usage:
    python bob_server_logged.py --port 5000 --logfile bob_log.txt
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from qiskit import transpile
from qiskit.qpy import load as qpy_load
from qiskit_aer import AerSimulator

from bob_server import recv_exact
from quantum_protocols import decode_result


def _write(logfile: str, entry: dict) -> None:
    """Append one JSON log entry (one line) to the log file."""
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def run_logged_server(host: str, port: int, logfile: str) -> None:
    """Accept one connection, decode the superdense circuit, log everything."""

    # Clear previous log
    with open(logfile, "w", encoding="utf-8") as fh:
        fh.write("")

    # Detect local IP for display
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    _write(logfile, {
        "type": "status",
        "message": f"Bob server listening on {local_ip}:{port}",
        "timestamp": time.time(),
    })

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(5)

            _write(logfile, {
                "type": "waiting",
                "message": "Waiting for Alice to connect...",
                "timestamp": time.time(),
            })

            try:
                conn, addr = srv.accept()
            except KeyboardInterrupt:
                _write(logfile, {"type": "status", "message": "Stopped.", "timestamp": time.time()})
                return
    except OSError as _bind_exc:
        _write(logfile, {
            "type": "error",
            "message": (f"Cannot bind to port {port}: {_bind_exc}. "
                        "Another process may already be using this port. "
                        "Click 'Stop Server' then 'Start Listening' again."),
            "timestamp": time.time(),
        })
        return

    _write(logfile, {
        "type": "connected",
        "message": f"Connection from {addr[0]}:{addr[1]}",
        "timestamp": time.time(),
    })

    try:
        # Peek at first 6 bytes to detect SECURE flag
        first_bytes = recv_exact(conn, 6)

        if first_bytes[:6] == b"SECURE":
            _write(logfile, {
                "type": "info",
                "message": "Secure mode (E91 encrypted)",
                "timestamp": time.time(),
            })
            from bob_server import handle_secure
            decoded = handle_secure(conn)

        else:
            # Standard superdense coding
            length_prefix = first_bytes[:4]
            extra = first_bytes[4:]
            payload_len = struct.unpack(">I", length_prefix)[0]

            _write(logfile, {
                "type": "info",
                "message": f"Receiving {payload_len} QPY bytes...",
                "timestamp": time.time(),
            })

            remaining = recv_exact(conn, payload_len - len(extra))
            payload = extra + remaining

            _write(logfile, {
                "type": "info",
                "message": f"Received {len(payload)} bytes. Deserializing circuit...",
                "timestamp": time.time(),
            })

            circuits = qpy_load(io.BytesIO(payload))
            circuit = circuits[0]

            _write(logfile, {
                "type": "info",
                "message": (f"Circuit: {circuit.num_qubits} qubits, "
                            f"{circuit.num_clbits} classical bits"),
                "timestamp": time.time(),
            })

            simulator = AerSimulator()
            compiled = transpile(circuit, simulator)
            job = simulator.run(compiled, shots=1024)
            counts = job.result().get_counts()

            _write(logfile, {
                "type": "info",
                "message": f"Counts: {dict(sorted(counts.items()))}",
                "timestamp": time.time(),
            })

            decoded = decode_result(counts, with_qec=False)

        _write(logfile, {
            "type": "decoded",
            "message": f"Decoded message: '{decoded}'",
            "decoded": decoded,
            "timestamp": time.time(),
        })

        conn.sendall(f"DECODED:{decoded}\n".encode("utf-8"))

        _write(logfile, {
            "type": "success",
            "message": f"SUCCESS — '{decoded}' received and acknowledged.",
            "decoded": decoded,
            "timestamp": time.time(),
        })

    except Exception as exc:
        _write(logfile, {
            "type": "error",
            "message": f"Error: {exc}",
            "timestamp": time.time(),
        })
        try:
            conn.sendall(f"ERROR:{exc}\n".encode("utf-8"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    _write(logfile, {
        "type": "done",
        "message": "Server session complete.",
        "timestamp": time.time(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QEC Dense — Bob server with file-based logging"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--logfile", default="bob_log.txt")
    parser.add_argument("--once", action="store_true", default=True)
    args = parser.parse_args()
    run_logged_server(args.host, args.port, args.logfile)


if __name__ == "__main__":
    main()
