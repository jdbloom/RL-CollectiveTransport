# Known Non-Determinism

Documents any non-determinism that remains after `DETERMINISM_ENABLED: true` is set.
Updated alongside `rl_code/src/determinism.py` and `tests/test_determinism_integration.py`.

---

## Helper: `apply_determinism_settings`

The helper in `rl_code/src/determinism.py` is **bit-exact on MPS**, verified by the
unit test `tests/test_agent/test_determinism.py::test_determinism_bit_exact_same_seed`.

Two independent 10-step smoke trainings with the same seed and
`DETERMINISM_ENABLED=true` produce **identical** agent parameter tensors
(`torch.equal()` true for every weight and bias in the Q-network).

Backend: PyTorch 2.3.1 / macOS 14 / Apple M-series (MPS).

---

## Production Wiring: `Main.py`

Verified by the integration test
`tests/test_determinism_integration.py::TestDeterminismIntegration::test_deterministic_runs_match`.

Two independent Main.py subprocess invocations with the same seed and
`DETERMINISM_ENABLED=true` in the YAML config produce **bit-exact HDF5 outputs**
(all episode attrs compared with `numpy.array_equal`) on MPS (Apple Silicon).

Attributes compared per episode: `reward_per_robot`, `gsp_output_std`,
`gsp_pred_target_corr`.

Max absolute diff observed on MPS (PyTorch 2.3.1): **0.0** (full bit-equality
on this backend; `numpy.array_equal` passed with no fallback needed in test runs).

The MPS ULP tolerance in the test is set to `< 1e-9` as a practical bound in
case a future PyTorch version introduces ULP-level drift. If that threshold is
ever triggered, the max diff magnitude will be logged in the test output and
should be documented here.

### Pre-fix status (feat/phase4-determinism, RL-CT @ 470c670)

The helper existed but was **never called from Main.py**. Despite `DETERMINISM_ENABLED: true`
in the YAML, all RNGs (numpy global, torch, random, action-noise, replay buffer sampling)
drew from un-seeded globals. The integration test
`test_determinism_integration.py::test_deterministic_runs_match` **fails** on that branch
with divergences of up to `6.4e-02` on `gsp_pred_target_corr` — consistent with the W1a
observation of ep 0 head_corr differing across reps (-0.0028 / -0.0129 / -0.0052).

---

## CUBLAS workspace env var timing

**Constraint:** `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` must be
set **before** `import torch` in the process entry point. The environment
variable is read once when the CUDA workspace is initialised; setting it after
torch is imported has no effect on CUDA.

**Fix in place:** Main.py sets this env var at the very top of the file
(before any torch import) when `DETERMINISM_ENABLED: true` is detected via
the `_early_read_determinism()` helper, which reads the YAML using `yaml.safe_load`
before any RL imports happen.

For MPS and CPU runs this constraint does not apply — those backends do not
use the CUBLAS workspace.

---

## Thread-reduction-order non-determinism (H2 from ARGoS investigation)

**Problem:** Under concurrent load (multiple worker processes on the same host),
the order in which BLAS reduction threads complete can differ between runs even
with the same seed, causing ULP-level weight divergence in GEMM operations.

**Fix:** `apply_determinism_settings` calls `torch.set_num_threads(1)` and the
early-init block sets `OMP_NUM_THREADS=1` before any BLAS library is loaded.
This closes both the OpenMP and ATen thread pools, eliminating thread-scheduling
as a source of non-determinism.

---

## DataLoader workers

Any `torch.utils.data.DataLoader` constructed while `DETERMINISM_ENABLED=true`
should use `num_workers=0` to avoid inter-process RNG state divergence. The
current codebase (RL-CollectiveTransport) does not use DataLoader, so no
change is needed here. If DataLoader is introduced in future work, enforce
`num_workers=0` in the DataLoader constructor when `self.determinism_enabled`
is True.

---

## MPS-specific notes

`torch.use_deterministic_algorithms(True)` is supported on MPS as of PyTorch
2.x. Deterministic backward passes through ReLU and Adam optimizer steps
produce bit-exact outputs on MPS in PyTorch 2.3.1. If a future PyTorch
version introduces MPS non-determinism for any operation used by this
codebase, `use_deterministic_algorithms(True)` will raise a `RuntimeError`
at the offending op, giving a loud failure rather than silent variance.
