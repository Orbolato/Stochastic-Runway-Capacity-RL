# Model-free reinforcement learning for runway capacity management under uncertainty

[![reproducibility](https://github.com/Orbolato/Stochastic-Runway-Capacity-RL/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/Orbolato/Stochastic-Runway-Capacity-RL/actions/workflows/reproducibility.yml)

Code, processed data, retained PPO models, and archived results for the paper:

> Carvalho, L.O., Murça, M.C.R. (2025). **A stochastic model-free reinforcement learning framework for optimizing runway capacity management under uncertainty.** *Transportation Research Part A: Policy and Practice*, 200, 104620. https://doi.org/10.1016/j.tra.2025.104620

Publication page: https://orbolato.github.io/publications/runway-capacity-reinforcement-learning/

The paper formulates runway capacity management at São Paulo/Guarulhos International Airport (SBGR) as a Markov Decision Process that jointly solves runway configuration management (RCM) and arrival–departure runway balancing (ADRB). It compares Maskable PPO (Stable-Baselines3 Contrib) with a Deep Q-Network implemented from scratch in PyTorch under two families of training approaches:

| Approach | Training data | Testing data |
|---|---|---|
| DB Approach 1 | Observed demand + METAR, 487 days (70%) | 209 days |
| DB Approach 2 | Observed demand + METAR, 689 days (99%) | 7 days |
| FB Approach 1 | Scheduled demand + deterministic TAF, one model per test day | 7 days |
| FB Approach 2 | Stochastic scenarios generated from schedules and TAF, one model per test day | 7 days |
| Baseline | Served demand and actual runway configuration (flow managers' policy) | 696 days |

A genetic algorithm (GA) with full knowledge of each day provides an upper benchmark.

## Reproducibility scope

This repository distinguishes three reproducibility levels:

1. **Artifact reproduction.** The committed per-day return tables regenerate Figs. 13–19 without retraining. CI checks this path on every push and pull request.
2. **Experiment rerunning.** The PPO, DQN, and GA code can be rerun from the committed processed inputs. The original RL and GA experiments did **not** use fixed random seeds, so rerunning is not expected to reproduce the archived numerical returns exactly.
3. **Raw-to-processed reconstruction.** The R preprocessing and RCCE-estimation code are preserved as historical methodology. The current `R_Codes/generate_sbgr_data.R` is **not** asserted to recreate the committed processed dataset byte-for-byte: the archived `data/data.csv` contains operational days 2023-04-24 through 2023-04-28, while the current R script explicitly excludes those five days. The repository history does not establish when that exclusion was introduced relative to the archived study inputs.

Accordingly, `data/data.csv` and `data/data_gru.Rda` should be treated as the authoritative archived processed inputs for experiment rerunning. The processed/aggregated datasets committed under `data/` are intentionally released publicly by the corresponding author as research artifacts for this paper. The underlying source traffic records listed in `data/README.md` remain excluded. See [`PROVENANCE.md`](PROVENANCE.md) for the precise release boundary, immutable artifact identifiers, and historical preprocessing note.

## Repository structure

```
├── .github/workflows/         Automated integrity and artifact-regeneration CI
├── R_Codes/                   Historical data-preprocessing and RCCE-estimation code (R)
│   ├── taf_decoder.R              Decodes raw TAFs into 15-min forecasts
│   ├── forecast_data_decoder_mod.R METAR/TAF decoding functions
│   ├── generate_sbgr_data.R       Historical SBGR traffic/METAR preprocessing (needs excluded source traffic data)
│   └── capacity_management.R      RCCE quantile regression (AMPL) and export of data.csv
├── optimization-models/       AMPL models for the quantile regression (Eqs. 1-2)
├── data/                      Public archived processed datasets and weather/TAF inputs
├── curves/                    RCCEs per weather condition, runway configuration, and percentile
├── ppo_data_based/            Maskable PPO, DB Approaches 1 and 2, greedy policy, and GA benchmark
├── ppo_forecast_based/        Maskable PPO, FB Approaches 1 and 2
├── dqn/                       Deep Q-Network, DB Approaches 1 and 2 and FB Approach 2
├── trained_models/ppo/        Retained PPO models associated with the study
├── results/                   Archived per-day rewards and Figs. 13-19 regeneration
├── tests/                     Repository integrity tests
├── CITATION.cff               Machine-readable citation metadata
└── PROVENANCE.md              Data/result provenance and reproducibility boundary
```

## Where each part of the paper is

| Paper | Code |
|---|---|
| RCCE estimation by quantile regression (Section 3.1, Eqs. 1-2, Fig. 12) | `R_Codes/capacity_management.R`, `optimization-models/linear_quantreg.mod`; Fig. 12 is plotted by `ppo_data_based/maskable_ppo_test.py` (`rcces_gru.eps`) |
| MDP: states, actions, reward (Section 3.2, Eqs. 7-13) | `ppo_data_based/improved_gym_capoptimizer.py` (PPO, DB), `ppo_forecast_based/improved_gym_capoptimizer_specific_std.py` (PPO, FB 1), `ppo_forecast_based/improved_gym_capoptimizer_specific.py` (PPO, FB 2), `dqn/improved_runway_env.py` (DQN, DB), `dqn/improved_runway_env_sp.py` (DQN, FB 2) |
| DQN with state and action as inputs (Fig. 3) | `dqn/improved_dqn_agent.py` (DB), `dqn/dqn_agent.py` (FB 2) |
| Stochastic scenario generation (Section 3.3, Eqs. 14-15) | `ppo_forecast_based/improved_gym_capoptimizer_specific.py`, `dqn/improved_runway_env_sp.py` |
| Error distributions of scheduled demand (Figs. 8-9) | `ppo_data_based/maskable_ppo_test.py` (`arr_dist.eps`, `dep_dist.eps`, `arr_variation.eps`, `dep_variation.eps`) |
| Confusion and transition matrices (Figs. 10-11) | `ppo_forecast_based/maskable_ppo_test.py` (`wc_error.eps`, `ws_error.eps`, `wc_trans.eps`, `ws_trans.eps`) |
| PPO, DB Approaches 1 and 2 | `ppo_data_based/maskable_ppo_test.py` |
| PPO, FB Approaches 1 and 2 | `ppo_forecast_based/maskable_ppo_test.py` |
| DQN, DB Approaches 1 and 2 | `dqn/dqn_cap_optimization.py` |
| DQN, FB Approach 2 | `dqn/dqn_cap_optimization_sp.py` |
| Baseline policy | `base_action()` / `base_step()` in the environment classes |
| GA benchmark | `ppo_data_based/runway_optimization_ga.py` (DEAP), run by `ppo_data_based/calculations.py` |
| Results (Figs. 13-19) | `results/returns_*.csv`, `results/make_result_figures.py` |

## Installation

Python 3.12 with the packages in `requirements.txt` (the versions used for the prepared experiment environment):

```bash
conda create -n rl-runway python=3.12
conda activate rl-runway
pip install -r requirements.txt
```

The R preprocessing uses R 4.4 with `plyr`, `dplyr`, `ggplot2`, `nloptr`, `lubridate`, and `RCurl`. The RCCE estimation also needs [AMPL](https://ampl.com) with the `rAMPL` package and a Gurobi license. Set `ampl_dir` at the top of `R_Codes/capacity_management.R` to your AMPL installation before running that optional historical preprocessing stage.

For the lightweight repository checks and result-figure regeneration only, CI uses the smaller `requirements-ci.txt` environment.

## Usage

The archived processed data and RCCEs are already in `data/` and `curves/`; these are the intended inputs for rerunning the Python experiments.

### 1. Raw-to-processed preprocessing (R, historical/optional)

The source code for the preprocessing pipeline is preserved for methodology transparency. Run from the repository root, in this order:

```bash
Rscript R_Codes/taf_decoder.R            # data/new_taf_list.csv, data/std_taf.csv, data/taf_workspace.RData
Rscript R_Codes/generate_sbgr_data.R     # data/data_gru.Rda (requires excluded source traffic data)
Rscript R_Codes/capacity_management.R    # data/data.csv and curves/*.csv
```

**Important:** do not treat the current `generate_sbgr_data.R` as an exact reconstruction recipe for the committed `data.csv`/`data_gru.Rda`. The archived processed dataset contains operational days 24–28 April 2023, whereas the current script excludes them. This discrepancy is preserved and documented rather than silently changing historical data or code. See [`PROVENANCE.md`](PROVENANCE.md).

`capacity_management.R` can be run on its own from the included archived `data/data_gru.Rda` to recompute RCCEs. The Python scripts use the curves with p = 0.99 (VMC), 0.95 (IMC), and 0.90 (LIMC).

### 2. Experiments (Python)

The scripts were developed in Spyder and are organized in cells separated by `####` headers. Run them from their own folder (Spyder's default working directory), either cell by cell or in full:

```bash
cd ppo_data_based
python maskable_ppo_test.py      # PPO, DB Approaches 1 and 2
```

```bash
cd ppo_forecast_based
python maskable_ppo_test.py      # PPO, FB Approaches 1 and 2
```

```bash
cd dqn
python dqn_cap_optimization.py      # DQN, DB Approaches 1 and 2
python dqn_cap_optimization_sp.py   # DQN, FB Approach 2
```

Each script loads `data/` and `curves/`, encodes the features, builds the action sets, filters the congested days, defines the train/test days, and then trains and tests the agents. Trained models and figures are written into the `Results - ...` subfolders.

`ppo_data_based/calculations.py` computes the GA benchmark. It uses the variables created by `maskable_ppo_test.py` (data, test days, action sets, and greedy policy), so it must be run in the same Python session afterwards (for example, in Spyder's console or with `%run -i calculations.py` in IPython). For each test day it keeps the better of the GA solution and the greedy warm start.

Training budgets: each PPO model is trained with `model.learn(1_000_000)` (about 30 min per model on the original CPU workflow); each DQN agent runs 1,000,000 episodes. The paper describes one million training episodes for both algorithms; for Maskable PPO, the implementation budget is one million timesteps (about 10,400 days of 96 periods).

DB Approach 2 is incremental: the DB Approach 1 agent (PPO or DQN) continues training on the 689 days of DB Approach 2.

### 3. Retained PPO models

`trained_models/ppo/` contains the Maskable PPO model archives retained from the study (DB Approaches 1 and 2, and one model per test day for FB Approaches 1 and 2). They allow the retained PPO policies to be evaluated without retraining. For example:

```python
model = MaskablePPO.load('../trained_models/ppo/db_approach_1/ppo_model_db_1.zip', env=env)
```

DQN weights were not retained; the DQN implementation and archived per-day returns are preserved.

### 4. Archived results and figures

`results/results.txt` preserves the historical per-day total rewards from the experiments. The same values are organized in two tidy files:

- `results/returns_209_test_days.csv`: GA, PPO, DQN, and baseline for the 209 test days of DB Approach 1
- `results/returns_7_test_days.csv`: all methods and approaches for the 7 test days of DB/FB Approach 2

To redraw Figs. 13–19 in `results/figures/`:

```bash
python results/make_result_figures.py
```

DQN obtains exactly the same archived returns on the 7 test days with DB Approach 1 and FB Approach 2. These are preserved as separate historical experiment outputs rather than deduplicated.

No random seeds were fixed in the original experiments, so retraining the agents or rerunning the GA is expected to give returns that differ from the archived values. See [`results/README.md`](results/README.md) and [`PROVENANCE.md`](PROVENANCE.md).

## Quick repository verification

The verification path does **not** retrain the agents. It checks artifact integrity, the archived processed-data shape, the documented April preprocessing distinction, result tables, retained model archives, CFF metadata, and the public-package boundary:

```bash
pip install -r requirements-ci.txt
python -m compileall -q dqn ppo_data_based ppo_forecast_based results tests
python -m unittest discover -s tests -p 'test_*.py' -v
python results/make_result_figures.py
```

GitHub Actions performs the same class of checks automatically and regenerates all 28 PNG/EPS result-figure files in a temporary directory.

## Data availability

The processed data needed to run the Python experiments and RCCE estimation are included and intentionally released publicly in this repository; see [`data/README.md`](data/README.md). The underlying source traffic records (ANAC VRA/SIROS, Flightradar24-based trajectories, and DECEA TATIC records) are not redistributed. As stated in the paper, access to those source datasets may be requested from the corresponding author, subject to the applicable source-data access conditions.

## Citation

GitHub also exposes the machine-readable [`CITATION.cff`](CITATION.cff). Please cite the published article:

```bibtex
@article{carvalho2025runway,
  title   = {A stochastic model-free reinforcement learning framework for optimizing runway capacity management under uncertainty},
  author  = {Carvalho, Lucas Orbolato and Mur{\c{c}}a, Mayara Cond{\'e} Rocha},
  journal = {Transportation Research Part A: Policy and Practice},
  volume  = {200},
  pages   = {104620},
  year    = {2025},
  doi     = {10.1016/j.tra.2025.104620}
}
```

## License

No software or data license is asserted by this repository. Please follow the publication citation and the provenance/access conditions documented for the research materials.
