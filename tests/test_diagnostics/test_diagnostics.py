import json
from pathlib import Path

import pytest

from src.diagnostics import ExperimentLogger


def test_creates_temp_dir(tmp_path):
    logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
    assert "test_exp" in str(logger.log_dir)
    assert Path(logger.log_dir).is_dir()
    logger.close()


def test_creates_log_files(tmp_path):
    logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
    assert (Path(logger.log_dir) / "python.log").exists()
    logger.close()


def test_cleanup_on_success(tmp_path):
    logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
    log_dir = logger.log_dir
    logger.finish(success=True)
    assert not Path(log_dir).exists()


def test_preserve_on_failure(tmp_path):
    logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
    log_dir = logger.log_dir
    logger.finish(success=False, error_message="ARGoS crashed")
    assert Path(log_dir).exists()
    error_file = Path(log_dir) / "crash_dump" / "error.txt"
    assert error_file.exists()
    assert "ARGoS crashed" in error_file.read_text()


def test_get_logger_returns_named_logger(tmp_path):
    logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
    child = logger.get_logger("main")
    assert child.name == "stelaris.test_exp.main"
    child.info("hello from test")
    logger.close()
    log_content = (Path(logger.log_dir) / "python.log").read_text()
    assert "hello from test" in log_content


def test_write_crash_dump(tmp_path):
    logger = ExperimentLogger("test_exp", base_dir=str(tmp_path))
    logger.write_crash_dump(
        last_state={"step": 42, "reward": -1.5},
        error_message="segfault in argos",
    )
    crash_dir = Path(logger.log_dir) / "crash_dump"
    assert (crash_dir / "error.txt").exists()
    assert "segfault in argos" in (crash_dir / "error.txt").read_text()
    state = json.loads((crash_dir / "last_state.json").read_text())
    assert state["step"] == 42
    logger.close()
