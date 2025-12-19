# Anomaly Detection in Spacecraft Telemetry Data

**CSC 371: Machine Learning - Fall 2025 - Final Project**  

## Project Overview

This project implements an unsupervised ensemble machine learning approach for detecting anomalies in multivariate spacecraft telemetry data from NASA's Soil Moisture Active Passive (SMAP) satellite. The system combines three anomaly detection algorithms through majority voting to identify both point and contextual anomalies across nine telemetry channels.

## Dataset

- **Source**: NASA SMAP Telemetry Anomaly Dataset from Kaggle
- **Format**: `.npy` files, one per telemetry channel
- **Channels**: 9 anonymised channels (A, B, D, E, F, G, P, R, S, T) representing different spacecraft subsystems
- **Features**: 25 multivariate sensor readings per channel
- **Anomalies**: 55 manually labelled sequences (62% point anomalies, 38% contextual anomalies)
- **Structure**: Multivariate time series with each timestamp representing one minute of aggregated telemetry

## Methodology

### Feature Engineering
- **Temporal Windows**: 60-timestamp sliding windows with 20-timestamp overlap (40 shared between consecutive windows)
- **MiniRocketMultivariate**: 20,000 random convolutional kernels for feature extraction
- **Z-Score Normalisation**: StandardScaler for feature scaling

### Models Implemented
- **One-Class SVM**: Boundary-based anomaly detection
- **Isolation Forest**: Partition-based isolation of anomalies
- **Local Outlier Factor (LOF)**: Density-based outlier detection
- **Ensemble Model**: Majority voting (threshold ≥1) combining all three algorithms

### Threshold Optimisation
- Percentile-based approach (1st-30th percentiles)
- Prioritises recall over precision (missing anomalies carries greater operational risk)
- Channel-specific threshold selection

## Results

- **Average Recall**: 0.55 (ensemble outperforms individual models)
- **Average F1 Score**: 0.21
- **Best Channel**: D (precision: 0.29, recall: 0.61, F1: 0.39)
- **Highest Recall**: S (0.80) and G (0.78) despite low anomaly rates
- **Performance**: 2/10 channels achieve recall > 0.7

## Key Challenges Addressed

1. **Class Imbalance**: Anomalies represent 1.5%-22.6% of data across channels
2. **Temporal Dependencies**: Overlapping windows capture contextual patterns
3. **Anonymisation**: Channel-agnostic approach generalises across subsystems
4. **Recall-Precision Trade-off**: Designed for operational spacecraft monitoring where false negatives are critical

## Repository Structure

```
├── model_two.ipynb              # Main analysis notebook with ensemble implementation
├── labeled_anomalies.csv        # Ground truth anomaly metadata
├── raw_train/                   # Training data (CSV format)
├── raw_test/                    # Testing data (CSV format)
├── data/                        # Windowed data by channel
├── transformed_data/            # MiniRocket-transformed features
├── paper/Writeup.pdf            # Academic paper
└── README.md
```

## Usage

1. **Data Preprocessing**: Run cells 1-10 in `model_two.ipynb` to process raw data
2. **Feature Engineering**: Execute MiniRocket transformation and scaling (cells 11-12)
3. **Model Training**: Run individual model cells (One-Class SVM, Isolation Forest, LOF)
4. **Ensemble Evaluation**: Execute ensemble cell to generate combined predictions
5. **Visualisation**: Run summary cells to generate performance plots and tables

## Visualisations

The notebook generates four key figures:
- **Model Comparison**: 4-panel comparison showing recall, F1, anomaly distribution, and overall performance
- **Performance Heatmap**: Metrics across all channels
- **Performance Metrics**: Bar charts comparing precision/recall/F1/FDR
- **Confusion Matrices**: Grid showing true/false positives/negatives per channel