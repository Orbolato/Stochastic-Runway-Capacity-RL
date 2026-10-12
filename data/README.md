# Data

All data refer to São Paulo/Guarulhos International Airport (SBGR) between
2022-01-02 06:00 UTC and 2023-12-30 05:45 UTC. Time is discretized into 15-min
periods, and each operational day runs from 06:00 UTC (D0) to 05:45 UTC (D+1),
giving 96 periods per day. The committed processed dataset contains **696 days /
66,816 periods**.

## Public processed-data release

The corresponding author has elected to make the processed and aggregated study
datasets committed in this directory publicly available as research artifacts for
the paper. In particular, `data.csv` and `data_gru.Rda` contain aggregated
operational variables derived from multiple source datasets. Their inclusion in
this repository does **not** imply permission to redistribute the excluded
underlying source records listed below.

See [`../PROVENANCE.md`](../PROVENANCE.md) for the release boundary, immutable
artifact identifiers, and the documented historical preprocessing note.

## Historical preprocessing note

The committed `data.csv` is an archived study input and is treated as authoritative
for rerunning the Python experiments in this repository. An integrity audit found
that it contains operational days **2023-04-24 through 2023-04-28**, each with 96
periods. The current committed `R_Codes/generate_sbgr_data.R`, however, explicitly
excludes those five operational days.

The available repository history does not establish when or why that exclusion was
introduced relative to the archived processed dataset. For that reason, this
repository does **not** claim that rerunning the current raw-to-processed R pipeline
will recreate the committed `data.csv` or `data_gru.Rda` byte-for-byte. The
historical source code and the archived processed study inputs are both preserved
without silently modifying either one. See [`../PROVENANCE.md`](../PROVENANCE.md).

## Files

| File | Produced by / provenance | Used by | Description |
|---|---|---|---|
| `data.csv` | Archived processed study artifact; preprocessing methodology in `R_Codes/` | all Python experiments | Traffic and METAR features per 15-min period (main dataset) |
| `new_taf_list.csv` | `R_Codes/taf_decoder.R` | all Python experiments | Decoded TAF forecasts per 15-min period, including probabilistic (PROB/TEMPO) groups |
| `std_taf.csv` | `R_Codes/taf_decoder.R` | all Python experiments | "Standard" deterministic TAF scenario: rows of probabilistic groups replaced by the main forecast |
| `data_gru.Rda` | Archived processed study artifact; preprocessing methodology in `R_Codes/generate_sbgr_data.R` | `R_Codes/capacity_management.R` | R version of the main dataset, input of the RCCE estimation |
| `raw-data/taf.csv` | REDEMET (DECEA) | `R_Codes/taf_decoder.R` | Raw TAF messages for SBGR |
| `raw-data/metar_sbgr_2019_2023.Rda` | Iowa Environmental Mesonet | `R_Codes/generate_sbgr_data.R` | Decoded METARs for SBGR |

The Runway Configuration Capacity Envelopes (RCCEs) are in [`../curves`](../curves).

## `data.csv` columns

Counts are numbers of flights in the 15-min period starting at `time`.

| Column | Description |
|---|---|
| `time` | Start of the period (UTC) |
| `scheduled_arrivals_40` | Scheduled arrivals (ANAC SIROS), shifted to the 40-NM cylinder entry time by the unimpeded transit time (10th percentile of the 40 NM → landing time) |
| `arrival_demand_40` | Observed arrival demand: flights entering the 40-NM cylinder (A40) |
| `actual_arrivals` | Actual landings |
| `base_arrivals_40` | Served arrivals: landings shifted back to the 40-NM time by the unimpeded transit time |
| `*_100` | Same as the `_40` columns for the 100-NM cylinder (not used in the paper) |
| `*_edmus`, `*_vunox`, `*_zares`, `*_buxuk` | Same counts split by arrival fix (east, north, west, south; not used in the paper) |
| `scheduled_departures` | Scheduled departures (ANAC SIROS) |
| `departure_demand` | Observed departure demand: pushbacks (TATIC) |
| `actual_departures` | Actual takeoffs (TATIC) |
| `base_departures` | Served departures: takeoffs shifted back by the unimpeded taxi time (10th percentile of pushback → takeoff) |
| `configuration` | Runway configuration used by most landings in the period (`10R\|10L` or `28L\|28R`) |
| `base_configuration` | Same, aligned with the served arrivals (actual runway configuration of the baseline policy) |
| `sep_*` | Constant placeholder (not used) |
| `vsby`, `ceiling` | Visibility (statute miles) and ceiling (ft) from the latest METAR |
| `ifr`, `lifr` | IMC and LIMC flags; weather condition = `ifr + lifr` (0 VMC, 1 IMC, 2 LIMC) |
| `drct`, `sknt` | Wind direction (degrees) and speed (knots) from the latest METAR |
| `headwind_10`, `crosswind_10` | Head- and crosswind components (knots) for runways 10R/10L (heading 73°) |
| `wind_state` | 0 both configurations available, 1 only 10R\|10L, 2 only 28L\|28R, 3 none (tailwind > 5 kt or crosswind > 20 kt) |
| `day` | Operational day (`YYYY-M-D`) |

In the paper (Table 2), *scheduled* = `scheduled_*`, *observed* = `arrival_demand_40` / `departure_demand`,
*served* = `base_arrivals_40` / `base_departures`, and *actual* = `actual_*`.

## TAF files (`new_taf_list.csv`, `std_taf.csv`)

| Column | Description |
|---|---|
| `time`, `day` | Period start (UTC) and operational day |
| `probability` | Probability of the forecast group (1 for the main forecast) |
| `wind_direction_degrees`, `wind_speed_knt` | Forecast wind |
| `visibility_nm`, `ceiling_ft` | Forecast visibility and ceiling |
| `weather_condition` | 0 VMC, 1 IMC, 2 LIMC |
| `wind_state`, `headwind_10`, `crosswind_10` | As in `data.csv` |

## Source records not included

The source traffic records used by `R_Codes/generate_sbgr_data.R` are intentionally
not distributed in this repository because of their size and source-data licensing
or access conditions:

- `raw-data/traffic-data/vra.csv` and `siros_1.csv` … `siros_6.csv`: ANAC VRA and SIROS flight records
- `raw-data/traffic-data/KPI08_2022.rda`, `KPI08_2023.rda`: arrival trajectories at the 40/100-NM cylinders (Flightradar24-based)
- `raw-data/traffic-data/KPI17_v2/`: departure records (Flightradar24-based)
- `raw-data/traffic-data/BD_TF_Raw_v2/`: TATIC departure milestones (DECEA)
- `taf_workspace.RData`: R workspace saved by `taf_decoder.R` (regenerated by running it)

As stated in the paper, the excluded source datasets are available from the
corresponding author upon reasonable request, subject to the applicable access
conditions. The committed processed files are enough to run every Python
experiment and the RCCE estimation.

## Integrity

CI checks that `data.csv` contains 66,816 rows corresponding to 696 operational
days with 96 periods per day; confirms the archived April 24–28 inclusion while
also checking that the current preprocessing script contains the documented
exclusion statements; and verifies the exact committed bytes of the principal data
artifacts using their Git blob IDs. See
[`../tests/test_repository_integrity.py`](../tests/test_repository_integrity.py).
