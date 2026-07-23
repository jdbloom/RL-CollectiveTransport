"""Arm C Stage 0 — rod-task config threading tests (real fixtures, no mocks).

Covers the Python half of the rod slice:
  * make_config defaults preserve the legacy cylinder task (default-off).
  * resolve_rod_geometry validation: every invalid knob combination fails
    loud; the valid rod config derives GATE_MIN = CONSTRICTION_RATIO *
    ROD_LENGTH under the rotate-to-fit constraint ROD_WIDTH < gap <
    ROD_LENGTH.
  * generate_argos_xml end-to-end: config dict -> generate_argos.py CLI ->
    template substitution -> generated .argos XML attributes (parsed back
    with ElementTree — the same file ARGoS consumes).
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


# ---------------------------------------------------------------------------
# default-off: the cylinder task is untouched
# ---------------------------------------------------------------------------

class TestDefaultOff:
    def test_make_config_defaults(self):
        cfg = _base_config("rodcfg_defaults")
        assert cfg["OBJECT_SHAPE"] == "cylinder"
        assert cfg["ROD_LENGTH"] == 3.0
        assert cfg["ROD_WIDTH"] == 0.5
        assert cfg["CONSTRICTION_RATIO"] is None
        assert cfg["GATE_MIN"] == 4

    def test_resolve_noop_on_cylinder(self):
        cfg = _base_config("rodcfg_noop")
        before = dict(cfg)
        out = rbe.resolve_rod_geometry(cfg)
        assert out == before, "cylinder config must pass through untouched"

    def test_resolve_noop_when_shape_key_absent(self):
        """Configs that predate the rod knobs (no OBJECT_SHAPE key) pass."""
        cfg = _base_config("rodcfg_absent")
        for key in ("OBJECT_SHAPE", "ROD_LENGTH", "ROD_WIDTH",
                    "CONSTRICTION_RATIO"):
            del cfg[key]
        before = dict(cfg)
        assert rbe.resolve_rod_geometry(cfg) == before

    def test_yaml_roundtrip_none_ratio(self, tmp_path):
        """write_yaml_config turns None into the string "None"; a config
        cloned from a written agent_config.yml must still resolve as
        disengaged (the test-eval clone path for cylinder parents)."""
        import yaml
        cfg = _base_config("rodcfg_rt")
        path = tmp_path / "agent_config.yml"
        rbe.write_yaml_config(cfg, str(path))
        with open(path) as f:
            cloned = yaml.safe_load(f)
        assert cloned["CONSTRICTION_RATIO"] == "None"  # the YAML artifact
        out = rbe.resolve_rod_geometry(dict(cloned))
        assert out["GATE_MIN"] == cloned["GATE_MIN"]  # untouched, no raise

    def test_cylinder_xml_defaults(self, cleanup_argos):
        cfg = _base_config(
            "rodcfg_cyl", ARGOS_FILE_NAME="collectiveRlTransport_rodcfg_cyl.argos")
        cleanup_argos.append(cfg["ARGOS_FILE_NAME"])
        rbe.generate_argos_xml(cfg)
        attrs = _loop_fn_attrs(cfg["ARGOS_FILE_NAME"])
        assert attrs["object_shape"] == "cylinder"
        assert float(attrs["rod_length"]) == pytest.approx(3.0)
        assert float(attrs["rod_width"]) == pytest.approx(0.5)
        assert float(attrs["gate_minimum"]) == pytest.approx(4.0)
        assert attrs["use_prisms"] == "0"


# ---------------------------------------------------------------------------
# resolve_rod_geometry: fail-loud validation
# ---------------------------------------------------------------------------

class TestResolveValidation:
    def test_ratio_with_cylinder_raises(self):
        cfg = _base_config("rodcfg_v1", CONSTRICTION_RATIO=0.7)
        with pytest.raises(ValueError, match="CONSTRICTION_RATIO"):
            rbe.resolve_rod_geometry(cfg)

    def test_unknown_shape_raises(self):
        cfg = _base_config("rodcfg_v2", OBJECT_SHAPE="ellipsoid")
        with pytest.raises(ValueError, match="OBJECT_SHAPE"):
            rbe.resolve_rod_geometry(cfg)

    def test_rod_without_gate_raises(self):
        cfg = _base_config("rodcfg_v3", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=0.7, USE_GATE=0)
        with pytest.raises(ValueError, match="USE_GATE"):
            rbe.resolve_rod_geometry(cfg)

    def test_rod_with_curriculum_raises(self):
        cfg = _base_config("rodcfg_v4", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=0.7, GATE_CURRICULUM=1)
        with pytest.raises(ValueError, match="GATE_CURRICULUM"):
            rbe.resolve_rod_geometry(cfg)

    def test_rod_without_ratio_raises(self):
        cfg = _base_config("rodcfg_v5", OBJECT_SHAPE="rod")
        with pytest.raises(ValueError, match="CONSTRICTION_RATIO"):
            rbe.resolve_rod_geometry(cfg)

    def test_ratio_one_raises(self):
        """gap == rod_length: the rod does not fit lengthwise."""
        cfg = _base_config("rodcfg_v6", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=1.0)
        with pytest.raises(ValueError, match="rotate-to-fit"):
            rbe.resolve_rod_geometry(cfg)

    def test_gap_below_width_raises(self):
        """gap <= rod_width: even the aligned rod cannot pass."""
        cfg = _base_config("rodcfg_v7", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=0.1)  # gap 0.3 < width 0.5
        with pytest.raises(ValueError, match="rotate-to-fit"):
            rbe.resolve_rod_geometry(cfg)

    def test_gap_equal_width_raises(self):
        cfg = _base_config("rodcfg_v8", OBJECT_SHAPE="rod",
                           ROD_LENGTH=2.0, ROD_WIDTH=0.5,
                           CONSTRICTION_RATIO=0.25)  # gap 0.5 == width exactly
        with pytest.raises(ValueError, match="rotate-to-fit"):
            rbe.resolve_rod_geometry(cfg)

    def test_degenerate_rod_raises(self):
        cfg = _base_config("rodcfg_v9", OBJECT_SHAPE="rod",
                           ROD_LENGTH=0.5, ROD_WIDTH=0.5,
                           CONSTRICTION_RATIO=0.7)
        with pytest.raises(ValueError, match="rod geometry invalid"):
            rbe.resolve_rod_geometry(cfg)


# ---------------------------------------------------------------------------
# resolve_rod_geometry: the derivation
# ---------------------------------------------------------------------------

class TestResolveDerivation:
    @pytest.mark.parametrize("ratio,expected_gap", [
        (0.5, 1.5), (0.7, 2.1), (0.9, 2.7),
    ])
    def test_gate_min_derived(self, ratio, expected_gap):
        cfg = _base_config("rodcfg_d1", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=ratio)
        rbe.resolve_rod_geometry(cfg)
        assert cfg["GATE_MIN"] == pytest.approx(expected_gap)

    def test_tight_gap_warns(self, capsys):
        """ratio 0.3 -> gap 0.9 < width + 2*clearance (1.2): loud warning."""
        cfg = _base_config("rodcfg_d2", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=0.3)
        rbe.resolve_rod_geometry(cfg)
        assert cfg["GATE_MIN"] == pytest.approx(0.9)
        assert "[ROD][WARN]" in capsys.readouterr().out

    def test_engaged_log(self, capsys):
        cfg = _base_config("rodcfg_d3", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=0.7)
        rbe.resolve_rod_geometry(cfg)
        assert "[ROD] OBJECT_SHAPE=rod ENGAGED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# end-to-end: config -> generate_argos.py -> .argos XML
# ---------------------------------------------------------------------------

class TestXmlEndToEnd:
    def test_rod_xml(self, cleanup_argos):
        cfg = _base_config(
            "rodcfg_rod", OBJECT_SHAPE="rod", CONSTRICTION_RATIO=0.7,
            ARGOS_FILE_NAME="collectiveRlTransport_rodcfg_rod.argos")
        cleanup_argos.append(cfg["ARGOS_FILE_NAME"])
        rbe.generate_argos_xml(cfg)
        attrs = _loop_fn_attrs(cfg["ARGOS_FILE_NAME"])
        assert attrs["object_shape"] == "rod"
        assert float(attrs["rod_length"]) == pytest.approx(3.0)
        assert float(attrs["rod_width"]) == pytest.approx(0.5)
        # The constriction: gap = 0.7 * 3.0, threaded through gate_minimum.
        assert float(attrs["gate_minimum"]) == pytest.approx(2.1)
        assert attrs["use_gate"] == "1"
        assert attrs["gate_curriculum"] == "0"
        # The rod does NOT ride the composite-prism flags.
        assert attrs["use_prisms"] == "0"
        assert attrs["random_objs"] == "0"

    def test_rod_custom_geometry_xml(self, cleanup_argos):
        cfg = _base_config(
            "rodcfg_rod2", OBJECT_SHAPE="rod", ROD_LENGTH=2.4, ROD_WIDTH=0.8,
            CONSTRICTION_RATIO=0.5,
            ARGOS_FILE_NAME="collectiveRlTransport_rodcfg_rod2.argos")
        cleanup_argos.append(cfg["ARGOS_FILE_NAME"])
        rbe.generate_argos_xml(cfg)
        attrs = _loop_fn_attrs(cfg["ARGOS_FILE_NAME"])
        assert float(attrs["rod_length"]) == pytest.approx(2.4)
        assert float(attrs["rod_width"]) == pytest.approx(0.8)
        assert float(attrs["gate_minimum"]) == pytest.approx(1.2)

    def test_invalid_rod_config_never_writes_xml(self, cleanup_argos):
        name = "collectiveRlTransport_rodcfg_invalid.argos"
        cfg = _base_config("rodcfg_invalid", OBJECT_SHAPE="rod",
                           CONSTRICTION_RATIO=1.5, ARGOS_FILE_NAME=name)
        cleanup_argos.append(name)
        with pytest.raises(ValueError):
            rbe.generate_argos_xml(cfg)
        assert not os.path.exists(os.path.join(PROJECT_ROOT, "argos", name)), \
            "validation must run BEFORE the XML is generated"
