import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    """Manages ephemeral log directories for experiment runs."""

    def __init__(self, exp_name: str, base_dir: str = "/tmp/stelaris-runs"):
        self.exp_name = exp_name
        self.log_dir = os.path.join(base_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self._root_logger_name = f"stelaris.{exp_name}"
        self._logger = logging.getLogger(self._root_logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        log_file = os.path.join(self.log_dir, "python.log")
        self._file_handler = logging.FileHandler(log_file)
        self._file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        self._file_handler.setFormatter(formatter)
        self._logger.addHandler(self._file_handler)

        self._closed = False

    def get_logger(self, component: str) -> logging.Logger:
        return logging.getLogger(f"{self._root_logger_name}.{component}")

    def write_crash_dump(
        self, last_state: dict | None = None, error_message: str | None = None
    ):
        crash_dir = Path(self.log_dir) / "crash_dump"
        crash_dir.mkdir(parents=True, exist_ok=True)

        if error_message is not None:
            (crash_dir / "error.txt").write_text(
                f"time: {datetime.now(timezone.utc).isoformat()}\n"
                f"experiment: {self.exp_name}\n"
                f"error: {error_message}\n"
            )

        if last_state is not None:
            with open(crash_dir / "last_state.json", "w") as f:
                json.dump(last_state, f, indent=2)

    def finish(self, success: bool, error_message: str | None = None):
        if success:
            self.close()
            # Only remove files we created (python.log), not the whole directory
            # The runner may have placed argos.log here too
            for name in ("python.log",):
                path = os.path.join(self.log_dir, name)
                if os.path.exists(path):
                    os.remove(path)
            # Remove crash_dump if it exists (shouldn't on success, but be safe)
            crash_dir = os.path.join(self.log_dir, "crash_dump")
            if os.path.isdir(crash_dir):
                shutil.rmtree(crash_dir)
            # Remove the directory only if empty (runner's files may still be there)
            try:
                os.rmdir(self.log_dir)
            except OSError:
                pass  # Directory not empty — runner still has files there
        else:
            self.write_crash_dump(error_message=error_message)
            self.close()

    def close(self):
        if self._closed:
            return
        self._file_handler.close()
        self._logger.removeHandler(self._file_handler)
        self._closed = True
