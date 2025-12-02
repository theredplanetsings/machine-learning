# USGS E. coli Prediction Models - Recreation Project

This directory contains a recreation of the multiple linear regression models developed by the U.S. Geological Survey (USGS) to predict E. coli concentrations at recreational sites on the Great Lakes.

### 1. `model.py`
**Python module containing model classes**

This module provides two main classes that recreate the USGS models:

- **`HuntingtonEcoliModel`**: Recreates the Huntington Beach (Pennsylvania) E. coli prediction model
  - Predicts LOG10[E. coli CFU/100mL]
  - Utilises 5 predictor variables: Lake temperature, turbidity (LOG10), wave height (SQRT), lake level change, and rainfall (SQRT)
  - Trained on 1,011 observations from 2005-2018

- **`Beach6EcoliModel`**: Recreates the Beach6 (Ohio) E. coli prediction model
  - Predicts E. coli LOG10 concentrations
  - Utilises 7 predictor variables: Turbidity (LOG10), humidity, water temperature, bird counts, lake level change, wind speed, and rainfall (SQRT)
  - Trained on 463 observations from 2014-2019

**Key Features:**
- Data loading and preprocessing
- Feature transformation (LOG10, SQRT)
- Model training utilising scikit-learn's `LinearRegression`
- Model evaluation (R², RMSE, sensitivity, specificity, accuracy)
- Coefficient comparison with USGS models
- Helper function for printing model summaries

### 2. `model.ipynb`
**Jupyter notebook for model training and analysis**

This comprehensive notebook walks through the entire process of recreating the USGS models:

**Contents:**
1. **Setup and Imports** - Load required libraries and modules
2. **Huntington Beach Model**
   - Data exploration and visualisation
   - Feature engineering
   - Model training
   - Performance evaluation
   - Comparison with USGS results
3. **Beach6 Model**
   - Data exploration and visualisation
   - Feature engineering
   - Model training
   - Performance evaluation
   - Comparison with USGS results
4. **Model Comparison** - Side-by-side comparison of both models
5. **Key Findings and Conclusions**

**Visualisations Include:**
- Raw variable distributions
- Predicted vs. actual scatter plots
- Residual plots
- Feature importance charts
- Model performance comparisons

## Project Objectives

1. Recreate USGS E. coli prediction models utilising scikit-learn
2. Compare our coefficients with the original USGS models
3. Evaluate model performance utilising the same metrics as USGS
4. Visualise model predictions and residuals
5. Document findings and model quality

## Results Summary

### Huntington Beach Model
- **R^2 = 0.5487** (USGS: 0.5499) - Very close match
- **RMSE = 0.4433** (USGS: 0.4431) - Nearly identical
- **Accuracy = 85.99%** (USGS: 86.03%)
- **Coefficients match within 2%** for all predictors

### Beach6 Model
- **R^2 = 0.4770** (USGS: 0.4770) - Exact match
- **RMSE = 0.4841** (USGS: 0.4841) - Perfect match
- **Accuracy = 90.71%** (USGS: 90.71%)
- **Coefficients match within 1%** for all predictors

## How to Use

### Option 1: Run the Jupyter Notebook (Recommended)
```bash
# Open the notebook
jupyter notebook model.ipynb

# Or in VS Code, open the .ipynb file
```

The notebook will guide you through the entire analysis with visualisations and explanations.

### Option 2: Use the Python Module Directly
```python
from model import HuntingtonEcoliModel, Beach6EcoliModel

# Initialise and train Huntington model
huntington = HuntingtonEcoliModel()
data = huntington.load_data('models/Huntington_MAS_pkg/Huntington_2019_calibration_data.csv')
features = huntington.create_features(data)
huntington.fit(features)

# Evaluate
metrics = huntington.evaluate(features)
print(f"R² = {metrics['r_squared']:.4f}")

# Compare with USGS
comparison = huntington.compare_with_usgs()
print(comparison)
```

## Data Sources

**Original USGS Project:**
- [USGS ScienceBase Catalog](https://www.sciencebase.gov/catalog/item/5fe22dead34e30b9123f09b5)
- Title: "Data for multiple linear regression models for estimating Escherichia coli (E. coli) concentrations... Great Lakes NowCast, 2019"