"""Contract test: the RL-CT Agent.store_agent_transition override must ACCEPT
and FORWARD `phi` to super().

Main.py calls ``model.store_agent_transition(..., phi=sf_phi)`` unconditionally
(sf_phi is None unless GSP_SF_ENABLED). The GSP-RL base
``Actor.store_agent_transition`` accepts ``phi=None``, but the RL-CT ``Agent``
subclass OVERRIDES the method — so if the override drops ``phi``, EVERY training
run (IC included) crashes at the first store call with:
    TypeError: store_agent_transition() got an unexpected keyword argument 'phi'
This exact break shipped on the SF pin bump (stelaris #233) and blocked all
training until 2026-07-07. Pure source-inspection (no torch/ARGoS) so it runs in
CI and pins the SF integration so it cannot half-merge across the submodules.
"""
import pathlib
import re

AGENT_PY = pathlib.Path(__file__).resolve().parent.parent / "rl_code" / "src" / "agent.py"


def test_agent_override_accepts_and_forwards_phi():
    text = AGENT_PY.read_text()
    m = re.search(
        r"def store_agent_transition\(self,(?P<params>[^)]*)\):"
        r"(?P<body>.*?)return super\(\)\.store_agent_transition\((?P<superargs>[^)]*)\)",
        text,
        re.S,
    )
    assert m, "Agent.store_agent_transition override not found in agent.py"
    assert "phi" in m.group("params"), (
        "Agent.store_agent_transition override must accept `phi` — Main.py passes "
        "phi=sf_phi on every store call, so a missing param crashes all training."
    )
    assert "phi=phi" in m.group("superargs"), (
        "override must forward `phi=phi` to super().store_agent_transition, or the "
        "SF cumulant is silently dropped before it reaches the replay buffer."
    )
