# Known Non-Determinism

Documents any non-determinism that remains after `DETERMINISM_ENABLED: true` is set.
Updated alongside `rl_code/src/determinism.py`.

## Status: bit-exact on all tested backends

Verified on PyTorch 2.3.1 / macOS 14 / Apple M-series (MPS), running
`tests/test_agent/test_determinism.py`.

Two independent 10-step training runs with the same seed and
`DETERMINISM_ENABLED=true` produce **identical** agent parameter tensors
(`torch.equal()` true for every weight and bias). The unit test asserts this.

## CUBLAS workspace env var timing

**Constraint:** `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` must be
set **before** `import torch` in the process entry point. The environment
variable is read once when the CUDA workspace is initialised; setting it after
torch is imported has no effect on CUDA.

**Implication for Main.py:** When `DETERMINISM_ENABLED: true` is detected in
the config, `Main.py` must set the env var at the very top of the file (before
any torch import), or rely on the launcher script (e.g.
`run_exp_with_config.sh`) to export it in the shell environment before
starting the Python process.

For MPS and CPU runs this constraint does not apply — those backends do not
use the CUBLAS workspace.

## DataLoader workers

Any `torch.utils.data.DataLoader` constructed while `DETERMINISM_ENABLED=true`
should use `num_workers=0` to avoid inter-process RNG state divergence. The
current codebase (RL-CollectiveTransport) does not use DataLoader, so no
change is needed here. If DataLoader is introduced in future work, enforce
`num_workers=0` in the DataLoader constructor when `self.determinism_enabled`
is True.

## MPS-specific notes

`torch.use_deterministic_algorithms(True)` is supported on MPS as of PyTorch
2.x. Deterministic backward passes through ReLU and Adam optimizer steps
produce bit-exact outputs on MPS in PyTorch 2.3.1. If a future PyTorch
version introduces MPS non-determinism for any operation used by this
codebase, `use_deterministic_algorithms(True)` will raise a `RuntimeError`
at the offending op, giving a loud failure rather than silent variance.
