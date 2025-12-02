# Cancer Type Classification Using miRNA Expression Profiles

This project implements a machine learning pipeline to classify six different types of cancer based on microRNA (miRNA) expression data from The Cancer Genome Atlas (TCGA).

## Project Structure

```
├── src/              # Source code and notebooks
├── paper/            # Research paper and documentation
└── data/             # Dataset (not tracked in version control)
```

## Dataset

The project uses miRNA expression data for the following cancer types:
- Breast Invasive Carcinoma
- Kidney Renal Clear Cell Carcinoma
- Lung Adenocarcinoma
- Lung Squamous Cell Carcinoma
- Pancreatic Adenocarcinoma
- Uveal Melanoma

Download the dataset from [The Cancer Genome Atlas data](https://drive.google.com/file/d/1jPRdAAxf9GDv4TZu_hVbdOOmx3Adszjb/view?usp=drive_link) and extract it to a `data/` directory in the project root. The data directory is excluded from version control.

## Implementation

The machine learning pipeline includes:
- Data consolidation from multiple patient files
- Feature extraction and preprocessing
- Class imbalance handling through bootstrapping
- Multiple classifier evaluation (Random Forest, Gradient Boosting, SVM, Logistic Regression)
- Hyperparameter optimisation using grid search
- Performance evaluation and visualisation

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Results

The best performing model achieves over 95% accuracy in classifying cancer types based on miRNA expression profiles. Detailed results and analysis are available in the accompanying paper.

## Documentation

See `src/README.md` for execution instructions and `paper/Writeup.pdf` for the complete research paper.