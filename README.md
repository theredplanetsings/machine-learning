# Machine Learning Projects

A collection of applied machine learning projects spanning medical diagnostics, environmental forecasting, and agricultural analysis. Each project demonstrates practical implementations of supervised learning techniques addressing real-world classification and prediction challenges.

## Projects

### Cancer Type Classification Using miRNA Expression Profiles

**Location:** `cancer-id/`

A multi-class classification system that identifies six cancer types from microRNA expression data sourced from The Cancer Genome Atlas (TCGA). The pipeline processes patient miRNA profiles and employs ensemble methods to achieve accurate cancer type discrimination.

**Cancer Types:**
- Breast Invasive Carcinoma
- Kidney Renal Clear Cell Carcinoma
- Lung Adenocarcinoma
- Lung Squamous Cell Carcinoma
- Pancreatic Adenocarcinoma
- Uveal Melanoma

**Technical Approach:**
- Data consolidation from multiple patient files
- Bootstrap resampling for class imbalance mitigation
- Hyperparameter optimisation via grid search
- Comparative evaluation of Random Forest, Gradient Boosting, Support Vector Machines, and Logistic Regression

**Performance:** The optimal model achieves over 95% classification accuracy on test data.

**Documentation:** Detailed methodology and results available in `cancer-id/paper/Writeup.pdf`

---

### E. coli Concentration Prediction for Great Lakes Recreation Sites

**Location:** `e-coli-forecast/`

Recreation of U.S. Geological Survey (USGS) multiple linear regression models for predicting E. coli concentrations at recreational beaches on the Great Lakes. The project validates the original USGS models using scikit-learn implementations.

**Models Implemented:**
1. **Huntington Beach Model (Pennsylvania)** - Trained on 1,011 observations (2005-2018)
   - Predictors: lake temperature, turbidity, wave height, lake level change, rainfall
   - Performance: R² = 0.5487, RMSE = 0.4433, Accuracy = 85.99%

2. **Beach6 Model (Ohio)** - Trained on 463 observations (2014-2019)
   - Predictors: turbidity, humidity, water temperature, bird counts, lake level change, wind speed, rainfall
   - Performance: R² = 0.4770, RMSE = 0.4841, Accuracy = 90.71%

**Technical Approach:**
- Feature transformation (LOG10, SQRT) matching USGS methodology
- Statistical validation against original USGS coefficients (within 1-2% match)
- Performance metrics comparison with published USGS results

**Original Research:** [USGS ScienceBase Catalog](https://www.sciencebase.gov/catalog/item/5fe22dead34e30b9123f09b5)

---

### Farmland Characteristics Identification from Aerial Imagery

**Location:** `id-farmland-characteristics/`

Computer vision analysis of the Agriculture Vision 2021 dataset to detect and classify farmland characteristics from aerial imagery. The project implements three separate analysis pipelines for identifying distinct agricultural features.

**Analysis Tasks:**
- **Nutrient Deficiency Detection** - Identification of crop nutrient deficiencies
- **Water Presence Analysis** - Detection of standing water or irrigation patterns
- **Weed Coverage Assessment** - Classification of weed-infested areas

**Technical Approach:**
- Deep learning models for image segmentation and classification
- Feature extraction from multi-spectral aerial imagery
- Independent analysis pipelines for each characteristic type

**Documentation:** Complete analysis and findings in `id-farmland-characteristics/paper/Writeup.pdf`

---

## Technologies

**Core Libraries:**
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** scikit-learn
- **Deep Learning:** PyTorch
- **Visualisation:** Matplotlib, Seaborn
- **Development:** Jupyter Notebook

**Techniques Demonstrated:**
- Supervised classification (multi-class, binary)
- Regression analysis
- Ensemble methods (Random Forest, Gradient Boosting)
- Support Vector Machines
- Hyperparameter optimisation
- Class imbalance handling
- Feature engineering and transformation
- Model validation and comparison

## Repository Structure

```
machine-learning/
├── cancer-id/
│   ├── src/              # Implementation notebooks
│   ├── paper/            # Research documentation
│   └── README.md
├── e-coli-forecast/
│   ├── src/              # Model implementation
│   ├── models/           # USGS reference models and data
│   ├── paper/            # Analysis documentation
│   └── README.md
├── id-farmland-characteristics/
│   ├── src/              # Analysis notebooks
│   ├── paper/            # Research documentation
│   └── README.md
└── README.md
```

## Getting Started

Each project directory contains its own README with specific setup instructions, data requirements, and usage guidelines. Projects are self-contained and can be explored independently.

**General Requirements:**
- Python 3.8 or higher
- Jupyter Notebook
- Standard scientific Python stack (see individual project requirements)

## Data Access

External datasets are required for each project and are not included in this repository. Download links and instructions are provided in the respective project README files.

## Licence

This project is licenced under the MIT Licence. See the `LICENSE` file for details.
