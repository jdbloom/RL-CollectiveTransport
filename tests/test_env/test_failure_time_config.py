"""latest_failure_time config->XML threading tests (real fixtures, no mocks).

Covers the inert passthrough added for the De Hauwere solo-vs-team eval:
  * default config -> latest_failure_time="1000" in the generated XML
    (byte-identical to the previously hardcoded template value).
  * LATEST_FAILURE_TIME=0 override lands in the XML.
  * solo-eval combo (MAX_NUM_ROBOT_FAILURES=3, CHANCE_FAILURE=1.0,
    LATEST_FAILURE_TIME=0) -> deterministic failures at spawn, all three
    attributes present in the XML ARGoS consumes.
"""

import os
import sys
import xml.etree.ElementTree as ET

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import run_baseline_experiments as rbe  # noqa: E402


def _base_config(exp_name, **overrides):
    """Real production config via make_config (not a hand-rolled dict)."""
    cfg = rbe.make_config(
        exp_name=exp_name, gsp=False, neighbors=False, num_obstacles=0,
        use_gate=1, gate_curriculum=0, use_prisms=0, port=55599,
        num_episodes=2)
    cfg.update(overrides)
    return cfg


def _loop_fn_attrs(argos_file_name):
    path = os.path.join(PROJECT_ROOT, "argos", argos_file_name)
    tree = ET.parse(path)
    node = tree.getroot().find("loop_functions")
    assert node is not None, "loop_functions node missing from generated XML"
    return node.attrib


@pytest.fixture
def cleanup_argos():
    """Remove generated .argos files after each test (real files, argos/)."""
    names = []
    yield names
    for name in names:
        path = os.path.join(PROJECT_ROOT, "argos", name)
        if os.path.exists(path):
            os.remove(path)


class TestFailureTimeXml:
    def test_default_config_xml(self, cleanup_argos):
        """Default 1000 == the previously hardcoded template value."""
        cfg = _base_config(
            "failtime_default",
            ARGOS_FILE_NAME="collectiveRlTransport_failtime_default.argos")
        cleanup_argos.append(cfg["ARGOS_FILE_NAME"])
        assert cfg["LATEST_FAILURE_TIME"] == 1000
        rbe.generate_argos_xml(cfg)
        attrs = _loop_fn_attrs(cfg["ARGOS_FILE_NAME"])
        assert attrs["latest_failure_time"] == "1000"

    def test_zero_override_xml(self, cleanup_argos):
        cfg = _base_config(
            "failtime_zero",
            ARGOS_FILE_NAME="collectiveRlTransport_failtime_zero.argos")
        cfg["LATEST_FAILURE_TIME"] = 0
        cleanup_argos.append(cfg["ARGOS_FILE_NAME"])
        rbe.generate_argos_xml(cfg)
        attrs = _loop_fn_attrs(cfg["ARGOS_FILE_NAME"])
        assert attrs["latest_failure_time"] == "0"

    def test_solo_eval_combo_xml(self, cleanup_argos):
        """De Hauwere solo-eval: 3 failures, certain, at spawn."""
        cfg = _base_config(
            "failtime_solo",
            ARGOS_FILE_NAME="collectiveRlTransport_failtime_solo.argos")
        cfg["MAX_NUM_ROBOT_FAILURES"] = 3
        cfg["CHANCE_FAILURE"] = 1.0
        cfg["LATEST_FAILURE_TIME"] = 0
        cleanup_argos.append(cfg["ARGOS_FILE_NAME"])
        rbe.generate_argos_xml(cfg)
        attrs = _loop_fn_attrs(cfg["ARGOS_FILE_NAME"])
        assert attrs["max_robot_failures"] == "3"
        assert attrs["chance_failure"] == "1.0"
        assert attrs["latest_failure_time"] == "0"
