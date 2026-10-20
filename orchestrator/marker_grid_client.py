"""Reliable, idempotent UDP client for the marker-grid Raspberry Pi node."""

import json
import socket
import time
import uuid


class MarkerGridClient:
    MODES = {"off", "static", "blink"}

    def __init__(self, host, port, timeout=0.25, attempts=3):
        self.address = (host, port)
        self.timeout = timeout
        self.attempts = attempts

    def status(self):
        return self._request("status")

    def set_mode(self, mode, tile=None):
        if mode not in self.MODES:
            raise ValueError("marker-grid mode must be off, static, or blink")
        fields = {"mode": mode}
        if tile is not None:
            fields["tile"] = list(tile)
        return self._request("set_mode", **fields)

    def wait_until_ready(self, timeout=15.0):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                return self.status()
            except (OSError, RuntimeError) as error:
                last_error = error
                time.sleep(0.2)
        raise TimeoutError(
            f"marker-grid node at {self.address[0]}:{self.address[1]} "
            f"did not become ready: {last_error}"
        )

    def _request(self, command, **fields):
        request_id = uuid.uuid4().hex
        request = {
            "version": 1,
            "request_id": request_id,
            "command": command,
            **fields,
        }
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
        last_error = None
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            udp_socket.settimeout(self.timeout)
            for _ in range(self.attempts):
                try:
                    udp_socket.sendto(payload, self.address)
                    response_payload, _ = udp_socket.recvfrom(65535)
                    response = json.loads(response_payload.decode("utf-8"))
                    if response.get("request_id") != request_id:
                        continue
                    if not response.get("ok"):
                        raise RuntimeError(
                            response.get("error", "marker-grid request failed")
                        )
                    return response
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    last_error = error
        raise TimeoutError(
            f"marker-grid request to {self.address[0]}:{self.address[1]} "
            f"was not acknowledged after {self.attempts} attempts: {last_error}"
        )
