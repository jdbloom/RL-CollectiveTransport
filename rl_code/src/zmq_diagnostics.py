"""Enhanced ZMQ diagnostics wrapper.

Logs all send/recv operations with timestamps, detects stalls,
checks if ARGoS process is alive, tracks message rates.
"""

import logging
import os
import time
import threading


class DiagnosticSocket:
    """Transparent wrapper around a ZMQ socket with full diagnostics."""

    def __init__(self, socket, exp_name, logger=None, argos_pid=None,
                 stall_threshold_s=300):
        self._socket = socket
        self._exp_name = exp_name
        self._stall_threshold = stall_threshold_s
        self._argos_pid = argos_pid
        self._msg_count = 0
        self._recv_count = 0
        self._send_count = 0
        self._episode = 0
        self._last_recv_time = time.time()
        self._last_send_time = time.time()
        self._in_recv = False
        self._start_time = time.time()

        self._log = logger or logging.getLogger(f"stelaris.{exp_name}.zmq")

        self._log.info("ZMQ diagnostics started for %s (stall=%ds, argos_pid=%s)",
                       exp_name, stall_threshold_s, argos_pid)

        self._watchdog_running = True
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    def _check_argos_alive(self):
        if self._argos_pid is None:
            return True
        try:
            os.kill(self._argos_pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _watchdog_loop(self):
        while self._watchdog_running:
            time.sleep(10)
            if self._in_recv:
                wait_time = time.time() - self._last_recv_time
                if wait_time > self._stall_threshold:
                    argos_alive = self._check_argos_alive()
                    self._log.critical(
                        "STALL: recv blocked %ds | ep=%d msg=#%d | argos_alive=%s",
                        wait_time, self._episode, self._msg_count, argos_alive
                    )
                elif wait_time > 60:
                    self._log.warning(
                        "SLOW: recv waiting %ds | ep=%d msg=#%d",
                        wait_time, self._episode, self._msg_count
                    )

    def set_episode(self, ep):
        self._episode = ep
        self._log.info("Episode %d started | total_msgs=%d", ep, self._msg_count)

    def recv(self, *args, **kwargs):
        self._in_recv = True
        self._last_recv_time = time.time()
        self._msg_count += 1
        self._recv_count += 1
        result = self._socket.recv(*args, **kwargs)
        self._in_recv = False
        recv_time = time.time() - self._last_recv_time
        if recv_time > 5.0:
            self._log.warning("SLOW recv: %.1fs, %d bytes", recv_time, len(result))
        return result

    def recv_multipart(self, *args, **kwargs):
        self._in_recv = True
        self._last_recv_time = time.time()
        self._msg_count += 1
        self._recv_count += 1
        result = self._socket.recv_multipart(*args, **kwargs)
        self._in_recv = False
        recv_time = time.time() - self._last_recv_time
        total_bytes = sum(len(m) for m in result)
        if recv_time > 5.0:
            self._log.warning(
                "SLOW recv_multipart: %.1fs, %d parts, %d bytes",
                recv_time, len(result), total_bytes
            )
        return result

    def send(self, data, *args, **kwargs):
        self._msg_count += 1
        self._send_count += 1
        start = time.time()
        result = self._socket.send(data, *args, **kwargs)
        send_time = time.time() - start
        if send_time > 1.0:
            size = len(data) if isinstance(data, (bytes, bytearray)) else "?"
            self._log.warning("SLOW send: %.1fs, %s bytes", send_time, size)
        self._last_send_time = time.time()
        return result

    def send_multipart(self, data, *args, **kwargs):
        self._msg_count += 1
        self._send_count += 1
        start = time.time()
        result = self._socket.send_multipart(data, *args, **kwargs)
        send_time = time.time() - start
        if send_time > 1.0:
            self._log.warning("SLOW send_multipart: %.1fs", send_time)
        self._last_send_time = time.time()
        return result

    def bind(self, *args, **kwargs):
        return self._socket.bind(*args, **kwargs)

    def close(self):
        self._watchdog_running = False
        elapsed = time.time() - self._start_time
        self._log.info(
            "Socket closed | %d msgs (%d recv, %d send) in %.0fs | ep=%d",
            self._msg_count, self._recv_count, self._send_count,
            elapsed, self._episode
        )

    def __getattr__(self, name):
        return getattr(self._socket, name)
