# ReflectScore Benchmark Report

## Overview

ReflectScore measures how different reflection strategies affect hallucination, grounding, refusal behavior, and latency.

## Leaderboard

| System | Hallucination Rate | Grounding | Refusal | Backfire | Mean Latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.220 | 1.000 | 0.267 | - | 5.413 |
| confidence_reflection | 0.180 | 0.800 | 0.400 | 0.000 | 6.105 |
| cross_agent | 0.180 | 0.933 | 0.400 | 0.020 | 21.794 |
| forced_reflection | 0.180 | 0.933 | 0.467 | 0.020 | 40.088 |
| self_reflection | 0.200 | 1.000 | 0.333 | 0.000 | 19.276 |
| verifier_reflection | 0.220 | 0.933 | 0.267 | 0.020 | 40.130 |

## Confidence Intervals

- **baseline**: hallucination CI [0.1, 0.34], grounding CI [1.0, 1.0], refusal CI [0.058823529411764705, 0.5007352941176464], correction CI -, backfire CI -
- **confidence_reflection**: hallucination CI [0.08, 0.2804999999999996], grounding CI [0.5714285714285714, 1.0], refusal CI [0.15779352226720647, 0.6818779904306219], correction CI [0.0, 0.06], backfire CI [0.0, 0.0]
- **cross_agent**: hallucination CI [0.08, 0.28], grounding CI [0.7777777777777778, 1.0], refusal CI [0.15384615384615385, 0.6666666666666666], correction CI [0.0, 0.1], backfire CI [0.0, 0.06]
- **forced_reflection**: hallucination CI [0.08, 0.28], grounding CI [0.7692307692307693, 1.0], refusal CI [0.2, 0.7272727272727273], correction CI [0.0, 0.06], backfire CI [0.0, 0.06]
- **self_reflection**: hallucination CI [0.1, 0.32], grounding CI [1.0, 1.0], refusal CI [0.1, 0.5833333333333334], correction CI [0.0, 0.06], backfire CI [0.0, 0.0]
- **verifier_reflection**: hallucination CI [0.12, 0.34], grounding CI [0.7777777777777778, 1.0], refusal CI [0.058823529411764705, 0.5], correction CI [0.0, 0.06], backfire CI [0.0, 0.06]

## Figures

![Leaderboard](D:/reflex/visualizations/leaderboard.png)
![Iteration Curve](D:/reflex/visualizations/iteration_curve.png)
![Failure Heatmap](D:/reflex/visualizations/failure_heatmap.png)
![Confidence Plot](D:/reflex/visualizations/confidence_plot.png)
