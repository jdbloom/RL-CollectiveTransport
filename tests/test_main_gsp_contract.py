"""Contract test for Main.py's store_gsp_transition calls.

Under the direct-MSE GSP training path (GSP-RL fix/gsp-direct-mse-training),
Main.py must store the LABEL (not the previous prediction) in the action
field of the replay buffer for ALL GSP variants. This test asserts the call
shape via static inspection — it's not a runtime test, but it catches
regressions in the call signature without needing ARGoS.

See Stelaris docs/research/2026-04-13-gsp-information-collapse-analysis.md
for the full rationale.
"""

import pathlib
import re


MAIN_PY = pathlib.Path(__file__).resolve().parent.parent / "rl_code" / "Main.py"


def test_store_gsp_transition_passes_label_as_action_argument():
    """Every store_gsp_transition call in Main.py must pass label-related arg as 2nd.

    The 2nd positional arg of store_gsp_transition is the action field. Under
    direct-MSE training, the GSP predictor is trained via MSE against the label
    stored in this field. If any call site passes the previous prediction
    (`old_heading_gsp[i]`, `action`, etc.) instead of `label`, the predictor
    trains against its own old output — no supervision signal.
    """
    text = MAIN_PY.read_text()
    # Match any store_gsp_transition(...) call and capture its entire arg list.
    # Use re.DOTALL in case the call spans multiple lines.
    calls = re.findall(r'store_gsp_transition\s*\(([^)]*)\)', text, re.DOTALL)
    assert len(calls) >= 3, (
        f"expected at least 3 store_gsp_transition calls (one per branch in the "
        f"independent/shared/attention block), got {len(calls)}"
    )
    violations = []
    for i, call in enumerate(calls):
        args = [a.strip() for a in call.split(",")]
        assert len(args) >= 2, f"call {i}: expected at least 2 args, got {args}"
        second_arg = args[1]
        if "label" not in second_arg:
            violations.append((i, second_arg))
    assert not violations, (
        f"{len(violations)} store_gsp_transition call(s) do not pass `label` as "
        f"the 2nd arg (action field); violations: {violations}. Under direct-MSE "
        f"GSP training the action field must carry the supervised target."
    )
