# Continued Process Verification Analytics

Synthetic pharmaceutical manufacturing case study demonstrating how statistical process monitoring can identify process deterioration before specification failure.
## DOI

Zenodo archival release:

10.5281/zenodo.22552505

## V2.1 Statistical Validation Extension

Version 2.1 adds Monte Carlo validation of the CPV monitoring framework, including:

- stable-process false-signal assessment
- abrupt 1-SD, 2-SD, and 3-SD mean shifts
- gradual process drift
- increased process variability
- Phase-I baseline estimation sensitivity
- heavy-tailed distribution stress testing
- autocorrelation stress testing
- monitoring-horizon sensitivity
- change-point localization performance

The original 24-batch synthetic OSD dataset remains an illustrative worked case. The simulation study provides the primary statistical validation of the monitoring framework.

## Project Objective

Build a Python-based Continued Process Verification workflow that distinguishes:

- product specification compliance
- historical process stability
- statistical process signals
- investigation priorities

## Case Study Summary

The synthetic dataset contains 24 batches:

- B001-B018: Historical reference
- B019-B024: Monitoring period

All monitoring batches remained within the dissolution specification, while the analysis identified a sustained multivariate shift involving API PSD, compression force, tablet hardness, disintegration, and dissolution.

## Key Finding

Specification compliance does not necessarily mean the process remains consistent with its historical state.

The analysis prioritized API PSD and compression force for investigation while avoiding unsupported causal conclusions.

## Tools

- Python
- pandas
- matplotlib
- Statistical process monitoring
- Correlation analysis
- Z-score shift analysis

## Files

- `CPV_Project_V1_Final_Polished.ipynb` — full analysis
- `CPV_Dashboard_V1.png` — final dashboard
- `CPV_Synthetic_Source_Data.csv` — synthetic dataset
- `CPV_monitoring_summary.csv` — monitoring results
- `CPV_investigation_priority.csv` — investigation-priority summary

## Disclaimer

This project uses fully synthetic data for educational and portfolio purposes. It does not contain employer, commercial product, patient, or confidential manufacturing data.
