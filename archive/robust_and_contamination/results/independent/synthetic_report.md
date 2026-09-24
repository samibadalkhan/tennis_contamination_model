# Independent coherent synthetic recovery

Two seeds per generator: exploratory recovery checks, not a precise power assessment or bounded-lag gate. The aggregate RMSE table is invalid for inference when any fit diagnostic fails.

**Diagnostic verdict: FAIL — do not interpret the RMSE differences as evidence.**

Ordinary: 15/16 failed cells, max R-hat 1.535, 449 divergences.  
Robust: 16/16 failed cells, max R-hat 1.876, 382 divergences.

| Generator | Condition | Ordinary RMSE | Robust RMSE | Paired gain |
|---|---|---|---|---|
| rw | none | 0.1486 | 0.1258 | +0.0228 |
| rw | point_burst | 0.1161 | 0.1088 | +0.0073 |
| rw | tournament | 0.1913 | 0.1655 | +0.0258 |
| rw | permanent_jump | 0.1929 | 0.1879 | +0.0051 |
| arc | none | 0.1426 | 0.1501 | -0.0074 |
| arc | point_burst | 0.1711 | 0.1647 | +0.0064 |
| arc | tournament | 0.2339 | 0.1997 | +0.0342 |
| arc | permanent_jump | 0.1729 | 0.1628 | +0.0101 |
