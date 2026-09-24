# Post-hoc calibration control for the early real-point result

Grid was specified in the independent full-model protocol before this control. It was not registered in the early-real protocol; interpret only as a calibration diagnostic. The selected value is the maximum of the grid, so the optimum is not bracketed.

The 2012 grid selected temperature **2.0**, the grid maximum, so the optimum is not bracketed. Accuracy and ranking are unchanged; only probability calibration changes.

| 2013 mode | Ordinary | Temperature | Huber 2.5 | Temperature gain vs ordinary [tournament CI] | Temperature gain vs Huber [tournament CI] |
|---|---:|---:|---:|---:|---:|
| frozen | 0.58443 | 0.49257 | 0.56089 | +0.09186 [+0.01702, +0.17492] | +0.06832 [-0.00711, +0.15204] |
| online | 0.59594 | 0.50627 | 0.55744 | +0.08967 [+0.01846, +0.16869] | +0.05117 [-0.01967, +0.13297] |

Only four 2013 tournaments contribute to these intervals. The point estimate shows that ordinary-model overconfidence can more than explain the Huber match-log-loss advantage; the comparison with Huber remains statistically imprecise. This control does not change the registered point-log-loss result.
