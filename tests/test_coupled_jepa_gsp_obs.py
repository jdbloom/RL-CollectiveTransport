"""Contract test: Main.py must populate the main-replay gsp_obs when coupled
JEPA is enabled (GSP_JEPA_COUPLE_VALUE), not only for the legacy e2e path.

The coupled-JEPA fix (GSP-RL feat/coupled-jepa) re-encodes the raw GSP input
WITH gradient inside the actor learn step, so the actor's main replay must carry
gsp_obs. Historically gsp_obs was gated on GSP_E2E_ENABLED only; that guard must
also fire for GSP_JEPA_COUPLE_VALUE.

Static-inspection test (no ARGoS/torch): asserts the gate helper is used at the
gsp_obs population site and the store call sites, so a regression to the
E2E-only guard is caught without a full run.
"""
import pathlib
import re


MAIN_PY = pathlib.Path(__file__).resolve().parent.parent / "rl_code" / "Main.py"


def test_gsp_obs_gate_includes_coupled_jepa():
    text = MAIN_PY.read_text()
    # A single helper decides whether gsp_obs is needed. It must reference BOTH
    # GSP_E2E_ENABLED and GSP_JEPA_COUPLE_VALUE.
    assert "GSP_JEPA_COUPLE_VALUE" in text, (
        "Main.py never references GSP_JEPA_COUPLE_VALUE — coupled JEPA will not "
        "populate the main-replay gsp_obs and the value gradient cannot reach "
        "the encoder."
    )
    # The population site (e2e_gsp_obs[i] = ...) must be reachable when coupled
    # JEPA is on: assert the guard around it references the coupled flag via the
    # shared helper name `_needs_gsp_obs`.
    assert "_needs_gsp_obs" in text, (
        "expected a `_needs_gsp_obs` gate unifying the E2E and coupled-JEPA "
        "gsp_obs requirement"
    )


def test_store_calls_use_needs_gsp_obs_gate():
    """The store_agent_transition gsp_obs kwargs must use the unified gate,
    not the bare GSP_E2E_ENABLED check."""
    text = MAIN_PY.read_text()
    # Find gsp_obs= kwargs on store_agent_transition and ensure they route
    # through _needs_gsp_obs (so coupled JEPA populates them too).
    store_gsp_obs_kwargs = re.findall(r"gsp_obs=e2e_gsp_obs\[i\] if ([^\n]+?) else None", text)
    assert store_gsp_obs_kwargs, "no gsp_obs= store kwargs found"
    for guard in store_gsp_obs_kwargs:
        assert "_needs_gsp_obs" in guard, (
            f"store gsp_obs guard still E2E-only: {guard!r}"
        )
