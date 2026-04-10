import time
from unittest.mock import MagicMock

import pytest

from src.zmq_diagnostics import DiagnosticSocket


@pytest.fixture
def mock_socket():
    sock = MagicMock()
    sock.recv.return_value = b"\x00" * 64
    sock.recv_multipart.return_value = [b"\x00" * 32, b"\x00" * 32]
    sock.send.return_value = None
    sock.send_multipart.return_value = None
    return sock


@pytest.fixture
def diag_socket(mock_socket):
    ds = DiagnosticSocket(mock_socket, "test_exp", argos_pid=None)
    yield ds
    ds.close()


def test_recv_delegates_to_underlying_socket(diag_socket, mock_socket):
    result = diag_socket.recv()
    mock_socket.recv.assert_called_once()
    assert result == b"\x00" * 64


def test_recv_multipart_delegates(diag_socket, mock_socket):
    result = diag_socket.recv_multipart()
    mock_socket.recv_multipart.assert_called_once()
    assert len(result) == 2


def test_send_delegates(diag_socket, mock_socket):
    diag_socket.send(b"hello")
    mock_socket.send.assert_called_once_with(b"hello")


def test_send_multipart_delegates(diag_socket, mock_socket):
    parts = [b"a", b"b"]
    diag_socket.send_multipart(parts)
    mock_socket.send_multipart.assert_called_once_with(parts)


def test_message_counts_increment(diag_socket):
    diag_socket.recv()
    diag_socket.recv_multipart()
    diag_socket.send(b"x")
    diag_socket.send_multipart([b"y"])
    assert diag_socket._msg_count == 4
    assert diag_socket._recv_count == 2
    assert diag_socket._send_count == 2


def test_set_episode_updates_episode(diag_socket):
    diag_socket.set_episode(5)
    assert diag_socket._episode == 5


def test_getattr_proxies_to_underlying(diag_socket, mock_socket):
    mock_socket.setsockopt = MagicMock()
    diag_socket.setsockopt(1, 2)
    mock_socket.setsockopt.assert_called_once_with(1, 2)


def test_bind_delegates(diag_socket, mock_socket):
    diag_socket.bind("tcp://*:5555")
    mock_socket.bind.assert_called_once_with("tcp://*:5555")


def test_check_argos_alive_returns_true_when_no_pid():
    sock = MagicMock()
    ds = DiagnosticSocket(sock, "test_exp", argos_pid=None)
    assert ds._check_argos_alive() is True
    ds.close()


def test_close_stops_watchdog(diag_socket):
    diag_socket.close()
    assert diag_socket._watchdog_running is False
