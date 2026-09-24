# MCMC seed replication

This replication repeats every posterior used by the exploratory ATP 2025
online forecast with NUTS seed 1729. It keeps the original cohort, bimonthly
information cutoffs, four chains, 1,000 warmup and 1,000 retained draws,
`target_accept=0.9`, posterior prediction seeds, 16,384 burst simulations,
burst seed 4001, burst rate 0.5, and temperature 1.5 fixed.

The primary numerical comparison is the replicated online burst log loss and
its paired difference from the original-seed online burst forecast. The
substantive comparison is replicated burst versus replicated temperature.
Report all per-period R-hat, bulk ESS, and divergences. Do not replace or pool
the original chains. This is a post-hoc numerical stability analysis of an
already-opened ATP 2025 outcome set.
