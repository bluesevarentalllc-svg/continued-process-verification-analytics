# Continued Process Verification Analytics

Synthetic pharmaceutical manufacturing case study demonstrating how statistical process monitoring can identify process deterioration before specification failure.

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
