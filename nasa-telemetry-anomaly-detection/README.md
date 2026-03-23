# Spacecraft Telemetry Anomaly Detection

This project implements an unsupervised approach to detect anomalies in multivariate spacecraft telemetry. It uses overlapping temporal windows and MiniRocket feature transformation, then evaluates each channel with an ensemble of One-Class SVM, Isolation Forest, and Local Outlier Factor.

The dataset is the [NASA SMAP telemetry anomaly dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl), containing point and contextual anomalies across nine telemetry channels. The ensemble achieved an average recall of 0.55 and F1 of 0.21. Threshold optimization on decision scores prioritizes recall, which can introduce optimistic bias but reflects the cost of missed anomalies.

---

## Project Overview

Spacecraft telemetry data is high-dimensional, noisy, and dominated by nominal behavior. This project combines:

- Sliding-window time-series preprocessing
- MiniRocket-based feature transformation
- Ensemble of classical unsupervised anomaly detection models

---

## Repository Structure

```text
nasa-telemetry-anomaly-detection/
├── artifacts/                  # Saved scalers and MiniRocket artifacts
├── data/
│   └── raw/
│       └── npy_format/         # SMAP channel data and labels
├── results/
│   ├── data_frame_format/      # Metrics tables (CSV)
│   ├── figures/                # Generated plots
│   └── text_format/            # Metrics summaries (TXT)
└── src/
	 ├── preprocessing/          # Feature selection/transformation
	 ├── models/                 # One-Class SVM, Isolation Forest, LOF, ensemble
	 └── evaluation/             # Metrics and reporting
```

---

## Pipeline Summary

1. **Preprocessing**
	- Channel-wise extraction from raw telemetry (SMAP only).
	- Overlapping windows to preserve temporal context.
	- Z-score scaling for comparable feature scales.

2. **Feature Extraction**
	- MiniRocket transforms time-series windows into fixed-length features.

3. **Models**
	- One-Class SVM
	- Isolation Forest
	- Local Outlier Factor (LOF)
	- Ensemble with threshold-one voting to prioritize recall.

4. **Evaluation Metrics**
	- Precision, recall, and F-beta score on the anomalous class.

---

## Results

- Metrics are stored in `results/data_frame_format/`.
- Comparative plots are stored in `results/figures/`.

---

## Getting Started

```bash
pip install -r requirements.txt
```

1. Run preprocessing and feature extraction.
2. Train models in `src/models/`.
3. Evaluate using `src/evaluation/`.

---

## Future Work

Investigating deep learning baselines (LSTM and transformer models) to improve performance.