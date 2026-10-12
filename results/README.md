# Archived results

This directory preserves the per-day total rewards reported by the study and the plotting script used to regenerate Figs. 13–19.

## Files

- `results.txt`: historical experiment output retained from the study.
- `returns_209_test_days.csv`: GA, PPO DB Approach 1, DQN DB Approach 1, and baseline returns for the 209-day test set.
- `returns_7_test_days.csv`: all reported approaches on the seven-day comparison set.
- `make_result_figures.py`: redraws Figs. 13–19 from the two tidy CSV files.
- `figures/`: committed PNG and EPS renderings of those figures.

Run from the repository root:

```bash
python results/make_result_figures.py
```

The figure-regeneration path is deterministic with respect to the archived CSV values. It does not retrain the RL agents or rerun the GA.

## Reproducibility boundary

No random seeds were fixed in the original RL or GA experiments. Retraining or reoptimizing is therefore expected to produce slightly different numerical returns. The archived CSV files are the provenance records for the published result figures, while the experiment scripts provide the rerunnable methodology.

A repository audit also identified a historical preprocessing distinction that is preserved rather than silently modified: the archived `data/data.csv` contains operational days 24–28 April 2023, while the current `R_Codes/generate_sbgr_data.R` explicitly excludes those five days. The archived `returns_209_test_days.csv` contains a row labeled `2023-04-28`. The available repository history does not establish when the preprocessing exclusion was introduced relative to the archived processed dataset and experiment outputs, so no unsupported reconciliation is asserted. See [`../PROVENANCE.md`](../PROVENANCE.md) for the complete provenance statement.
