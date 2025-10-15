# hmm_postprocess.py
import numpy as np
import pandas as pd

def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _ck_index_to_log_emissions(row, limb_cols, tau=1.0, alpha=6.0, limb_weights=None):
    """
    Convert per-limb Cole–Kripke sleep_index into log-likelihoods for Sleep/Wake.
    Uses a logistic mapping around threshold tau with slope alpha.
    Returns (log_p_x_given_S, log_p_x_given_W)
    """
    if limb_weights is None:
        limb_weights = {i: 1.0 for i in range(1, len(limb_cols)+1)}

    log_pS = 0.0
    log_pW = 0.0
    for i, col in enumerate(limb_cols, start=1):
        si = row[col]
        pS = _logistic(alpha * (tau - si))
        pS = float(np.clip(pS, 1e-6, 1.0 - 1e-6))
        pW = 1.0 - pS
        w  = limb_weights.get(i, 1.0)
        log_pS += w * np.log(pS)
        log_pW += w * np.log(pW)
    return log_pS, log_pW

def _viterbi_decode(log_emissions: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray):
    """
    Viterbi for 2 states (0=Sleep, 1=Wake)
    log_emissions: (T, 2) array of log p(x_t|state)
    """
    T = log_emissions.shape[0]
    log_delta = np.zeros((T, 2), dtype=float)
    psi = np.zeros((T, 2), dtype=np.int32)

    # init
    log_delta[0] = log_pi + log_emissions[0]

    # recursion
    for t in range(1, T):
        for s in range(2):
            vals = log_delta[t-1] + log_A[:, s]
            psi[t, s] = int(np.argmax(vals))
            log_delta[t, s] = vals[psi[t, s]] + log_emissions[t, s]

    # backtrack
    states = np.zeros(T, dtype=np.int32)
    states[-1] = int(np.argmax(log_delta[-1]))
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    return states, log_delta

def apply_hmm_over_cole(output_df: pd.DataFrame,
                        num_limbs=4,
                        tau=1.0,
                        alpha=6.0,
                        p_sw=0.008,
                        p_ws=0.025,
                        limb_weights=None,
                        add_probs=True,
                        outfile=None) -> pd.DataFrame:
    """
    Expects DataFrame produced by your Cole-Kripke multi-limb pipeline
    (i.e., with columns 'Limb i sleep_index' for i in [1..num_limbs]).
    Adds columns:
      - hmm_state ('S'/'W')
      - hmm_sleep_prob (optional)
    Saves to outfile if provided.
    """
    limb_cols = [f"Limb {i} sleep_index" for i in range(1, num_limbs+1)]
    missing = [c for c in limb_cols if c not in output_df.columns]
    if missing:
        raise ValueError(f"Missing per-limb columns: {missing}")

    # emissions
    logs = output_df.apply(
        lambda row: _ck_index_to_log_emissions(row, limb_cols, tau=tau, alpha=alpha, limb_weights=limb_weights),
        axis=1, result_type='expand'
    )
    log_pS = logs[0].to_numpy()
    log_pW = logs[1].to_numpy()
    log_emissions = np.vstack([log_pS, log_pW]).T  # (T,2) as [S, W]

    # transitions / priors (log)
    A  = np.array([[1 - p_sw, p_sw],
                   [p_ws,     1 - p_ws]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    log_A  = np.log(np.clip(A,  1e-12, 1.0))
    log_pi = np.log(np.clip(pi, 1e-12, 1.0))

    states, log_delta = _viterbi_decode(log_emissions, log_A, log_pi)
    out = output_df.copy()
    out['hmm_state'] = np.where(states == 0, 'S', 'W')

    if add_probs:
        m = np.max(log_delta, axis=1, keepdims=True)
        sm = np.exp(log_delta - m)
        post = sm / sm.sum(axis=1, keepdims=True)
        out['hmm_sleep_prob'] = post[:, 0]  # prob of Sleep

    if outfile:
        out.to_csv(outfile, index=False)
        print(f"HMM-smoothed results saved to {outfile}")

    return out
