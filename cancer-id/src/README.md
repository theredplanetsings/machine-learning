# Source Code

This directory contains the implementation of the cancer classification pipeline.

## Files

- `model.ipynb` - Main Jupyter notebook containing the complete machine learning pipeline

## Prerequisites

Ensure you have downloaded and extracted The Cancer Genome Atlas dataset to a `data/` directory in the project root. The expected structure is:

```
../data/
├── Breast Invasive Carcinoma/
├── Kidney Renal Clear Cell Carcinoma/
├── Lung Adenocarcinoma/
├── Lung Squamous Cell Carcinoma/
├── Pancreatic Adenocarcinoma/
└── Uveal Melanoma/
```

Each cancer type directory contains patient subdirectories with miRNA quantification files (`*.mirbase21.mirnas.quantification.txt`).

## Running the Code

1. Install required dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. Open and run `model.ipynb` in Jupyter:
   ```bash
   jupyter notebook model.ipynb
   ```

3. Execute cells sequentially from top to bottom.

## Notebook Sections

1. **Import Libraries** - Load required packages
2. **Data Loading** - Read and consolidate patient data across cancer types
3. **Exploratory Data Analysis** - Visualise class distributions and feature characteristics
4. **Data Preprocessing** - Handle class imbalance through bootstrapping, encode labels, split data
5. **Dimensionality Visualisation** - PCA-based visualisation of cancer type separability
6. **Model Training** - Train and evaluate multiple classifiers with hyperparameter tuning
7. **Model Comparison** - Compare performance across all trained models
8. **Best Model Analysis** - Detailed analysis including confusion matrix and feature importance
9. **Summary** - Key findings and performance metrics
10. **Model Saving** - Serialise trained models for future use

## Configuration

Key parameters that can be adjusted in the notebook:

- `DATA_DIR` (cell 5) - Path to the data directory (default: `'./data'`)
- `test_size` (cell 14) - Proportion of data reserved for testing (default: `0.2`)
- `random_seed` (cell 1) - Random state for reproducibility (default: `42`)

## Output

The notebook generates:
- Performance metrics for all trained models
- Visualisations including class distributions, PCA plots, confusion matrices, and feature importance charts
- Trained model artefacts saved to `../models/` (optional, see final section)

## Notes

- The notebook uses a fixed random seed (`42`) for reproducibility
- All models are trained on bootstrapped data to address class imbalance
- Grid search is employed for hyperparameter optimisation
- Execution time varies depending on hardware; grid search operations may take several minutes