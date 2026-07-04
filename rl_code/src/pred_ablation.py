"""M2 — eval-time GSP prediction ablation.

The GSP prediction for every head variant funnels into ``next_heading_gsp[i]``
at a single injection site in Main.py. ``apply_pred_ablation`` transforms that
per-robot prediction vector at eval time according to ``GSP_EVAL_ABLATE_PRED``:

    none         -> identity (literal no-op; the default training path stays
                    bit-exact — the SAME object is returned, no allocation)
    zero         -> all-zeros, same shape/dtype (severs the prediction signal)
    shuffle      -> same multiset, order permuted under the seeded rng
                    (preserves the marginal distribution, destroys per-dim content)
    frozen_mean  -> the running mean of predictions accumulated across the episode
                    (replaces the live prediction with a constant of matched scale)

This is the M2 metric from the 2026-07-04 GSP actor-usage pre-reg: comparing
test-eval success across {none, zero, shuffle, frozen_mean} reveals whether the
actor *uses* the prediction, and in what form.

The helper is pure — it does not import Main.py. The running-mean accumulator is
externalized in ``RunningMeanState`` so Main.py owns the per-episode lifecycle
(reset at episode start), keeping this module stateless and testable in isolation.
"""

from __future__ import annotations

import numpy as np


class RunningMeanState:
    """Per-episode accumulator for the ``frozen_mean`` ablation mode.

    Main.py holds one instance per episode and resets it (constructs a fresh one)
    at each episode boundary so the mean never bleeds across episodes.
    """

    __slots__ = ("count", "_sum")

    def __init__(self) -> None:
        self.count: int = 0
        self._sum: np.ndarray | None = None

    def update_and_mean(self, pred_vec: np.ndarray) -> np.ndarray:
        """Fold ``pred_vec`` into the accumulator and return the running mean.

        The returned array has the same shape/dtype as ``pred_vec``.
        """
        if self._sum is None:
            self._sum = pred_vec.astype(np.float64, copy=True)
        else:
            self._sum = self._sum + pred_vec.astype(np.float64, copy=False)
        self.count += 1
        mean = self._sum / self.count
        return mean.astype(pred_vec.dtype, copy=False)


def apply_pred_ablation(pred_vec, mode, rng, running_mean_state):
    """Return the ablated per-robot GSP prediction for eval-time mode ``mode``.

    Parameters
    ----------
    pred_vec : np.ndarray
        The per-robot GSP prediction vector (shape (K,)) about to be written to
        ``next_heading_gsp[i]``.
    mode : str
        One of {'none', 'zero', 'shuffle', 'frozen_mean'}.
    rng : np.random.Generator
        Seeded generator used for the 'shuffle' permutation (deterministic).
    running_mean_state : RunningMeanState
        Per-episode accumulator used only by 'frozen_mean'.

    Notes
    -----
    'none' returns the SAME object (identity) so the default path is a literal
    no-op — required for bit-exact equivalence with un-instrumented training.
    The other modes never mutate ``pred_vec`` in place.
    """
    if mode == 'none':
        # Literal identity. Do NOT touch running_mean_state, do NOT allocate.
        return pred_vec
    if mode == 'zero':
        return np.zeros_like(pred_vec)
    if mode == 'shuffle':
        perm = rng.permutation(pred_vec.shape[0])
        return pred_vec[perm]
    if mode == 'frozen_mean':
        return running_mean_state.update_and_mean(pred_vec)
    raise ValueError(
        f"Unknown GSP_EVAL_ABLATE_PRED mode {mode!r}; "
        "expected one of 'none', 'zero', 'shuffle', 'frozen_mean'."
    )
