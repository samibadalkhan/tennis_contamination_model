"""Stage 4 (optional) -- two-state latent HMM over points.

Clean state emits Bernoulli(p_theta); contaminated state its own process, with
persistent transitions. EM responsibilities become per-point weights; eps is
the stationary distribution, episode length the relaxation time.

Intentionally unimplemented: build only after Stages 2-3.
"""
