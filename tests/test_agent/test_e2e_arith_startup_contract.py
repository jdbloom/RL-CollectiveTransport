"""Static source contracts on rl_code/Main.py for the
GSP_E2E_UNIFIED_TARGET_ARITH fail-loud startup line (same technique as
TestMainStartupContract in test_advantage_splice.py / \
test_batched_actor_forward.py — no ARGoS, no imports of Main.py).

Why the line must live in Main.py at all: GSP-RL emits its own ENGAGED/off
startup line from Hyperparameters.__init__, but to logger "stelaris.learn" —
which has NO handler in production. The run logger is stelaris.<exp_name>
with propagate=False, and logging.lastResort drops INFO, so the GSP-RL-side
line is silently discarded on every real run (it surfaces only under pytest
caplog). Main.py therefore emits the authoritative per-run python.log line,
keyed on the Actor's gate attribute gsp_e2e_unified_arith_engaged (single
condition source, set in Hyperparameters.__init__ as GSP_E2E_ENABLED and
GSP_E2E_UNIFIED_TARGET_ARITH) — the same pattern as the
GSP_SPLICE_ADVANTAGE_ONLY and BATCHED_ACTOR_FORWARD lines.
"""
import pathlib


def _main_text():
    return (pathlib.Path(__file__).resolve().parents[2]
            / "rl_code" / "Main.py").read_text()


class TestE2eArithStartupContract:

    def test_engaged_and_off_startup_lines_present(self):
        text = _main_text()
        assert "GSP_E2E_UNIFIED_TARGET_ARITH: ENGAGED" in text
        assert "GSP_E2E_UNIFIED_TARGET_ARITH: off" in text

    def test_log_keyed_on_actor_side_gate(self):
        text = _main_text()
        assert "gsp_e2e_unified_arith_engaged" in text, (
            "startup line must read the Actor's effective gate attribute, "
            "not re-derive the condition from raw config")

    def test_log_emitted_after_agent_construction(self):
        text = _main_text()
        construct = text.find("Agent.Agent(")
        log_line = text.find("GSP_E2E_UNIFIED_TARGET_ARITH: ENGAGED")
        assert construct != -1 and log_line != -1
        assert construct < log_line, (
            "the ENGAGED line reads the constructed Agent's gate attribute, "
            "so it must come after Agent construction")

    def test_missing_attribute_branch_warns_not_off(self):
        """A GSP-RL pin that predates the gate attribute must produce a loud
        UNKNOWN warning, never a false 'off' claim (fail-loud: fallbacks
        announce themselves)."""
        text = _main_text()
        assert "engagement state UNKNOWN" in text
        assert "gsp_e2e_unified_arith_engaged', None)" in text, (
            "the getattr default must be None (three-way: engaged / off / "
            "attribute-missing), not False — False silently remaps a "
            "pre-attribute pin to a confident 'off' line")
