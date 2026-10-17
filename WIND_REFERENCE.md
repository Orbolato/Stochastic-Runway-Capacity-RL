# Wind-direction reference convention

The SBGR runway/wind calculations in this repository use a **true-north reference frame**.

METAR and TAF wind directions are meteorological directions referenced to true north. When the historical preprocessing computes runway-relative wind components and the resulting wind-availability state, those directions are compared with the SBGR runway axis expressed in the same reference frame:

- runway 10 direction: **73° true**;
- reciprocal runway 28 direction: **253° true**.

The values 73°/253° are rounded runway-axis bearings used by the historical preprocessing. They are **not** the runway designator values (10/28 interpreted as 100°/280°), and they are not magnetic headings. Using true wind directions and true runway bearings in the same trigonometric calculation avoids a magnetic/true-north reference mismatch.

This convention applies to the METAR/TAF wind-component and runway-availability preprocessing in files such as `R_Codes/generate_sbgr_data.R` and `R_Codes/forecast_data_decoder_mod.R`. The Python reinforcement-learning environments consume the resulting processed wind state rather than performing a second heading conversion.

A later SBGR runway-configuration forecasting study uses more precise true bearings of approximately 73.69°/253.67°. This repository intentionally preserves the rounded 73°/253° values used by its historical preprocessing; this documentation does not modify the code, processed data, archived models, or published results.
