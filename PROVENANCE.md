# Provenance and reproducibility notes

This repository is the public research package associated with:

> Lucas Orbolato Carvalho and Mayara Condé Rocha Murça (2025), “A stochastic model-free reinforcement learning framework for optimizing runway capacity management under uncertainty,” *Transportation Research Part A: Policy and Practice*, 200, 104620. DOI: 10.1016/j.tra.2025.104620.

The initial public package was committed on 12 October 2026 as commit `fd5a5b4f87ae855bcca8ec0367f8b72569241fd3`. The hardening work added documentation, machine-checkable integrity tests, and CI without replacing the archived study data, results, capacity curves, or trained PPO model files.

## Publicly released processed study data

The corresponding author has elected to make the processed and aggregated study datasets committed under `data/` publicly available as research artifacts for this paper. In particular, `data/data.csv` and `data/data_gru.Rda` contain aggregated operational variables derived from several source datasets. Their inclusion here does **not** imply permission to redistribute the excluded underlying source records.

The public package includes:

- processed 15-minute SBGR traffic and METAR features (`data/data.csv` and `data/data_gru.Rda`);
- decoded TAF scenarios (`data/new_taf_list.csv` and `data/std_taf.csv`);
- the raw TAF extract used by the decoder (`data/raw-data/taf.csv`);
- the SBGR METAR archive used by preprocessing (`data/raw-data/metar_sbgr_2019_2023.Rda`);
- runway configuration capacity envelopes under `curves/`;
- archived per-day reward tables under `results/`;
- the Maskable PPO model archives retained from the study under `trained_models/ppo/`.

## Source records not redistributed

The following source records used by `R_Codes/generate_sbgr_data.R` are intentionally excluded from the public repository:

- ANAC VRA and SIROS flight records used during preprocessing;
- Flightradar24-based arrival and departure trajectory extracts;
- DECEA TATIC departure milestone records.

The exclusion is deliberate and is enforced by `.gitignore`, repository integrity tests, and CI. The public processed datasets are sufficient to run the Python experiments and to recompute the RCCEs without reconstructing these source records.

## Historical results and stochastic retraining

The committed files in `results/` are archived outputs from the study. No random seeds were fixed in the original RL or GA experiments. Consequently, retraining PPO or DQN, or rerunning the GA, is **not expected to reproduce the archived numerical returns exactly**. The repository therefore distinguishes:

1. **artifact reproduction** — regenerating Figs. 13–19 from the committed per-day return tables; and
2. **experiment rerunning** — retraining/reoptimizing from the processed data, where numerical variation is expected.

The retained PPO archives allow the original PPO policies to be evaluated without retraining. DQN model weights were not retained; the DQN contribution is preserved through its source code and archived per-day result tables.

## April 2023 historical-output note

`R_Codes/generate_sbgr_data.R` excludes operational days 24–28 April 2023 in the current public preprocessing path. The archived 209-day result table, however, contains a row labeled `2023-04-28`. Both facts are preserved intentionally: the result table is an historical study artifact and has not been edited to force agreement with the current cleaned preprocessing script.

The available repository history does not contain an immutable run manifest that establishes why this historical row differs from the current preprocessing exclusion. It would therefore be inappropriate to invent a reconciliation. CI explicitly verifies both conditions so that future edits cannot silently erase the discrepancy. This note should be read as provenance documentation, not as a claim that the current preprocessing script recreates the identical historical train/test sample.

## Integrity anchors

Git blob IDs below identify the exact bytes of the principal immutable study artifacts in the initial public package. `tests/test_repository_integrity.py` verifies these IDs directly from file bytes and also checks all 16 retained PPO archives.

| Artifact | Git blob ID |
|---|---|
| `data/data.csv` | `5660ea374ddd186dff838907a8b76ec20c0f0472` |
| `data/data_gru.Rda` | `9490d9594af504bc33ababa050997ef819bfcc20` |
| `data/new_taf_list.csv` | `61b4ead4316413a0d939700632657b6a7d02c8cd` |
| `data/std_taf.csv` | `f747e78347f78ab230dd593b58f4812e42a60fff` |
| `data/raw-data/metar_sbgr_2019_2023.Rda` | `280267e00f9c562c1ac0543dc711750b8307a217` |
| `data/raw-data/taf.csv` | `1f7b0e8a77b8ca6a38a58ca39b97fb1926e8b916` |
| `results/results.txt` | `f5f3d8a719f7f7e208a0fbc09d2d623096cef97b` |
| `results/returns_209_test_days.csv` | `95c4a05e05cf8985f117a62821f927d45d6d4471` |
| `results/returns_7_test_days.csv` | `aca9f98a98b0d9e3beca5df6ef50021f48ac79ca` |

The six production RCCE files used by the Python experiments (VMC p=0.99, IMC p=0.95, LIMC p=0.90 for both configurations) are also pinned in the integrity test.

## Automated verification scope

GitHub Actions performs a lightweight verification suitable for every push and pull request. It:

- compiles all Python source files;
- validates the exact bytes of the committed data/result/model anchors;
- checks the processed dataset shape (66,816 periods, 696 operational days, 96 periods per day);
- checks the archived result-table row counts (209 and 7 days);
- confirms that restricted source traffic records are absent;
- validates the CFF citation metadata;
- regenerates all 28 PNG/EPS files corresponding to Figs. 13–19 from the archived reward tables.

CI deliberately does **not** retrain the million-step/episode RL agents or rerun the GA. Those are computational experiments, and the original experiments were not seeded.

## Licensing

No software or data license is asserted by this repository. The publication citation and source-data provenance remain applicable to reuse of the research materials.
