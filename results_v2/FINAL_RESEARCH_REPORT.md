# NanoSwarm 2.0 — Experimental Results

## Experiment scope

This report is generated from the recorded trial data.
The full target matrix contains 7,200 trials:
5 population sizes × 4 densities × 3 failure probabilities ×
30 seeds × 4 policies.

## Data files

- trials.csv: individual trial records
- summary.json: experiment summary
- paired_analysis.json: existing paired analysis
- final_analysis.json: final integrity and analysis metadata
- paired_policy_comparisons.csv: paired policy comparisons
- completion_comparison.png: completion figure
- runtime_seconds_comparison.png: runtime figure
- failures_comparison.png: failure figure

## Interpretation

Policy comparisons use matched configuration and seed records.
Positive differences mean the tested policy has a higher metric
than greedy; whether that is desirable depends on the metric.

Bootstrap intervals and sign-flip tests are exploratory.
They do not establish general superiority, and multiple comparisons
must be considered before making confirmatory claims.

## Limitations

The experiment is computational simulation evidence, not physical
robot validation. Results depend on the implemented environment,
reward function, policy implementation, and parameter choices.

The meaning of hybrid must be defined from the actual source code.
Do not describe it as controller fusion unless the implementation
actually performs controller fusion.

Reported active-ratio values must not be called time-averaged
unless they were accumulated across simulation steps.
