"""Time-varying serve/return ability — the shared backbone of both arms.

Each player carries a serve skill s and a return skill r, each a scalar Gaussian
belief that drifts as a random walk between matches (variance sigma^2 per year).
A match gives two binomial observations of serve points won:

    k_i ~ Binomial(n_i, sigmoid(alpha_surface + s_i - r_j))    # i serving
    k_j ~ Binomial(n_j, sigmoid(alpha_surface + s_j - r_i))    # j serving

The **ordinary** arm fits this by online assumed-density (Kalman-style)
filtering: between matches diffuse the variance; at a match do one Laplace/EKF
Gaussian update per serve observation and split it back onto the server's serve
skill and the returner's return skill by their variance shares. The **robust**
arm (later) is the same filter with a robustified update that caps the influence
of a single surprising observation (a burst).

No PPL is available in this environment, so this hand-rolled filter *is* the
inference — faithful to the random-walk dynamics and O(1) per observation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from src import forecast


class OnlineAbility:
    def __init__(self, alpha_by_surface: dict, sigma: float = 0.2,
                 prior_var: float = 1.0, robust_c: float | None = None):
        self.alpha = alpha_by_surface
        self.alpha_default = float(np.mean(list(alpha_by_surface.values())))
        self.sigma2 = sigma ** 2
        self.prior_var = prior_var
        self.robust_c = robust_c          # None = ordinary; else Huber-style cap (robust arm)
        self.sm: dict = {}; self.sv: dict = {}
        self.rm: dict = {}; self.rv: dict = {}
        self.last: dict = {}

    # --- state -----------------------------------------------------------------
    def _ensure(self, p, t):
        if p not in self.sm:
            self.sm[p] = 0.0; self.sv[p] = self.prior_var
            self.rm[p] = 0.0; self.rv[p] = self.prior_var
            self.last[p] = t

    def _diffuse(self, p, t):
        dt = max((t - self.last[p]).days / 365.25, 0.0)
        self.sv[p] += self.sigma2 * dt
        self.rv[p] += self.sigma2 * dt
        self.last[p] = t

    def _alpha(self, surface):
        return self.alpha.get(surface, self.alpha_default)

    # --- forecast --------------------------------------------------------------
    def serve_prob(self, server, returner, surface):
        return float(expit(self._alpha(surface) + self.sm[server] - self.rm[returner]))

    def predict_p1_wins(self, row) -> float:
        pa = self.serve_prob(row.p1, row.p2, row.surface)
        pb = self.serve_prob(row.p2, row.p1, row.surface)
        return forecast.p_match(pa, pb, int(row.best_of))

    # --- update ----------------------------------------------------------------
    def _update_serve_obs(self, server, returner, k, n, surface):
        vs, vr = self.sv[server], self.rv[returner]
        s2 = vs + vr
        mu = self._alpha(surface) + self.sm[server] - self.rm[returner]
        p0 = float(expit(mu))
        H = n * p0 * (1 - p0)                       # observed information at mu
        g = k - n * p0                              # score (gradient of log-lik)
        if self.robust_c is not None:               # robust arm: cap the innovation
            lim = self.robust_c * np.sqrt(max(H, 1e-9))
            g = float(np.clip(g, -lim, lim))
        s2_post = 1.0 / (1.0 / s2 + H)
        dmu = s2_post * g
        fs, fr = vs / s2, vr / s2                    # variance shares of theta = s - r
        self.sm[server] += fs * dmu
        self.rm[returner] -= fr * dmu               # theta up => returner's return skill down
        red = s2 - s2_post
        self.sv[server] = max(vs - fs * fs * red, 1e-6)
        self.rv[returner] = max(vr - fr * fr * red, 1e-6)

    def process(self, row, forecast_first: bool = False):
        """Diffuse both players, optionally forecast (before update), then update.
        Returns P(p1 wins) if ``forecast_first`` else None."""
        t = row.date
        self._ensure(row.p1, t); self._ensure(row.p2, t)
        self._diffuse(row.p1, t); self._diffuse(row.p2, t)
        pred = self.predict_p1_wins(row) if forecast_first else None
        # two serve observations (winner p1 serving; loser p2 serving)
        self._update_serve_obs(row.p1, row.p2, int(row.p1_spw), int(row.p1_svpt), row.surface)
        self._update_serve_obs(row.p2, row.p1, int(row.p2_spw), int(row.p2_svpt), row.surface)
        return pred


def alpha_by_surface(matches) -> dict:
    """Per-surface serve-win intercept = logit(overall serve-win rate on surface).
    Computed from the given (training) matches only; skills are centered at 0."""
    import pandas as pd
    spw = matches.p1_spw + matches.p2_spw
    svpt = matches.p1_svpt + matches.p2_svpt
    df = pd.DataFrame({"surface": matches.surface, "spw": spw, "svpt": svpt})
    rate = df.groupby("surface").apply(lambda d: d.spw.sum() / d.svpt.sum(), include_groups=False)
    return {s: float(np.log(r / (1 - r))) for s, r in rate.items()}
