"""Regime models done properly: Gaussian HMM with point-in-time state inference.

TWO TRAPS THIS MODULE EXISTS TO AVOID
-------------------------------------
1. SMOOTHED STATES ARE LOOKAHEAD. The usual `model.predict(X)` returns Viterbi /
   smoothed states, which infer the state at time t using the WHOLE sequence --
   including the future. Every regime study that does this is contaminated. We
   use FILTERED probabilities: P(state_t | observations up to t).

2. FITTING ON THE FULL SAMPLE IS LOOKAHEAD. Even filtered states are tainted if
   the transition matrix and emission parameters were estimated on data that
   includes the future. We refit on an expanding window and only ever apply a
   model to data after its training cut.

The cost is that this is slow and the states are noisier than the pretty charts
in most regime papers. That noise is the honest picture.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _fit_hmm(X: np.ndarray, n_states: int, seed: int = 0):
    from hmmlearn.hmm import GaussianHMM

    m = GaussianHMM(n_components=n_states, covariance_type="diag",
                    n_iter=200, random_state=seed, tol=1e-3)
    m.fit(X)
    return m


def filtered_states(observations: pd.DataFrame, n_states: int = 3,
                    min_train: int = 504, refit_every: int = 63,
                    seed: int = 0) -> pd.DataFrame:
    """Point-in-time filtered state probabilities.

    Returns a frame of P(state | data up to t), one column per state, plus a
    `state` column with the most likely state. Rows before `min_train` are NaN.

    Refits every `refit_every` bars on an expanding window; between refits the
    existing model is applied forward, which is what you could actually have
    done in real time.
    """
    X = observations.to_numpy(dtype=float)
    idx = observations.index
    n = len(X)
    out = np.full((n, n_states), np.nan)

    model = None
    order = None
    for t in range(min_train, n):
        if model is None or (t - min_train) % refit_every == 0:
            try:
                model = _fit_hmm(X[:t], n_states, seed)
                # HMM state labels are arbitrary; sort by mean of the first
                # observation column so "state 0" means the same thing across
                # refits. Without this the states permute and the conditional
                # analysis becomes nonsense.
                order = np.argsort(model.means_[:, 0])
            except Exception as exc:  # noqa: BLE001
                log.warning("HMM refit failed at %s: %s", idx[t], exc)
                continue
        try:
            # score_samples returns posteriors over the SUPPLIED sequence only,
            # so passing data up to t gives a filtered estimate at t.
            _, post = model.score_samples(X[:t + 1])
            out[t] = post[-1][order]
        except Exception:  # noqa: BLE001
            continue

    df = pd.DataFrame(out, index=idx,
                      columns=[f"p_state{i}" for i in range(n_states)])
    df["state"] = df.filter(like="p_state").to_numpy().argmax(axis=1)
    df.loc[df.filter(like="p_state").isna().all(axis=1), "state"] = np.nan
    return df


def smoothed_states_UNSAFE(observations: pd.DataFrame, n_states: int = 3,
                           seed: int = 0) -> pd.Series:
    """Full-sample Viterbi states. LOOKAHEAD -- for comparison only.

    Provided so the size of the bias can be measured rather than assumed. Never
    build a signal on this.
    """
    m = _fit_hmm(observations.to_numpy(dtype=float), n_states, seed)
    order = np.argsort(m.means_[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    return pd.Series([remap[s] for s in m.predict(observations.to_numpy(dtype=float))],
                     index=observations.index)
