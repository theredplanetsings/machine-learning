"""
USGS E. coli Prediction Models Recreation

This python file recreates the multiple linear regression models developed by the U.S. Geological Survey
for predicting E. coli concentrations at recreational sites on the Great Lakes.

Two models are implemented:
1. Huntington Beach Model (Pennsylvania)
2. Beach6 Model (Ohio)

Both models utilise environmental and water-quality variables as predictors.

must be visible to hw1.ipynb (in same directory) to be run
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class HuntingtonEcoliModel:
    """
    Recreation of USGS Huntington Beach E. coli prediction model.
    
    Model equation:
    LOG10[EcoliAve_CFU] = -34.6615e-03 + 35.9571e-03*Lake_Temp_C + 
                          68.9557e-02*LOG10[Lake_Turb_NTRU] + 
                          19.4169e-02*SQUAREROOT[WaveHt_Ft] + 
                          37.8094e-02*LL_PreDay + 
                          45.3704e-02*SQUAREROOT[AirportRain48W_in]
    
    Target variable: LOG10[EcoliAve_CFU]
    Predictors: Lake_Temp_C, LOG10[Lake_Turb_NTRU], SQUAREROOT[WaveHt_Ft], 
                LL_PreDay, SQUAREROOT[AirportRain48W_in]
    """
    # the USGS model coefficients for comparison
    USGS_COEFFICIENTS = {
        'intercept': -0.0346615,
        'Lake_Temp_C': 0.0359571,
        'LOG10_Lake_Turb_NTRU': 0.689557,
        'SQRT_WaveHt_Ft': 0.194169,
        'LL_PreDay': 0.378094,
        'SQRT_AirportRain48W_in': 0.453704
    }
    USGS_METRICS = {
        'r_squared': 0.54985,
        'adj_r_squared': 0.54896,
        'rmse': 0.4431,
        'sensitivity': 0.43011,
        'specificity': 0.95747,
        'accuracy': 0.86026
    }
    def __init__(self):
        """Initialise the Huntington Beach model."""
        self.model = LinearRegression()
        self.feature_names = [
            'Lake_Temp_C', 
            'LOG10_Lake_Turb_NTRU', 
            'SQRT_WaveHt_Ft', 
            'LL_PreDay', 
            'SQRT_AirportRain48W_in'
        ]
        self.is_fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads and preprocess Huntington Beach data.
        Args:
            filepath: Path to the CSV data file 
        Returns:
            Preprocessed DataFrame
        """
        df = pd.read_csv(filepath)
        # parses date if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates transformed features for the model.
        Args:
            df: Raw dataframe with original variables
        Returns:
            DataFrame with transformed features
        """
        df_features = df.copy()
        # creates LOG10 of E. coli (target variable) + adds small constant to handle zeroes
        df_features['LOG10_EcoliAve_CFU'] = np.log10(df_features['EcoliAve_CFU'] + 0.1)
        # creates LOG10 of turbidity
        # adds small constant to handle zeroes
        df_features['LOG10_Lake_Turb_NTRU'] = np.log10(df_features['Lake_Turb_NTRU'] + 0.1)
        # creates SQRT of wave height
        df_features['SQRT_WaveHt_Ft'] = np.sqrt(df_features['WaveHt_Ft'])
        # LL_PreDay is already in the data
        # creates SQRT of rainfall
        df_features['SQRT_AirportRain48W_in'] = np.sqrt(df_features['AirportRain48W_in'])
        
        return df_features
    
    def fit(self, df: pd.DataFrame) -> 'HuntingtonEcoliModel':
        """
        Fits the linear regression model.
        Args:
            df: DataFrame with features (must already be transformed) 
        Returns:
            Self for method chaining
        """
        # prepares features and target
        X = df[self.feature_names]
        y = df['LOG10_EcoliAve_CFU']
        # fits the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new data.
        Args:
            df: DataFrame with features (must already be transformed)
        Returns:
            Array of predictions (LOG10 scale)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_names]
        return self.model.predict(X)
    
    def predict_concentration(self, df: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions and convert back to CFU/100mL scale.
        Args:
            df: DataFrame with features (must already be transformed) 
        Returns:
            Array of predictions (CFU/100mL scale)
        """
        log_predictions = self.predict(df)
        return 10 ** log_predictions
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates model performance.
        Args:
            df: DataFrame with features and target (must already be transformed)
        Returns:
            Dictionary of evaluation metrics
        """
        X = df[self.feature_names]
        y_true = df['LOG10_EcoliAve_CFU']
        y_pred = self.predict(df)
        n = len(y_true)
        p = len(self.feature_names)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # calculates sensitivity, specificity, accuracy using threshold of
        # LOG10(235) = 2.371 for E. coli standard
        threshold = np.log10(235)
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        true_negatives = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / n
        
        return {
            'r_squared': r2,
            'adj_r_squared': adj_r2,
            'rmse': rmse,
            'mae': mae,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'n_observations': n
        }
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Gets model coefficients.
        Returns:
            Dictionary of coefficient names and values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        coefs = {
            'intercept': self.model.intercept_
        }
        for name, coef in zip(self.feature_names, self.model.coef_):
            coefs[name] = coef
            
        return coefs
    
    def compare_with_usgs(self) -> pd.DataFrame:
        """
        Compares fitted coefficients with USGS model.
        Returns:
            DataFrame comparing coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        our_coefs = self.get_coefficients()
        
        comparison = []
        for key, usgs_val in self.USGS_COEFFICIENTS.items():
            our_val = our_coefs.get(key, np.nan)
            diff = our_val - usgs_val
            pct_diff = (diff / usgs_val * 100) if usgs_val != 0 else np.nan
            
            comparison.append({
                'coefficient': key,
                'usgs_value': usgs_val,
                'our_value': our_val,
                'difference': diff,
                'pct_difference': pct_diff
            })
        
        return pd.DataFrame(comparison)

class Beach6EcoliModel:
    """
    Our recreation of USGS Beach6 E. coli prediction model.
    
    Model equation:
    ECOLI_LOG10 = -10.1271e-01 + 68.3547e-02*LOG10[TURB_NTRU] + 
                  79.3788e-04*RHUM_PCT + 52.5621e-03*WTEMP_CEL + 
                  18.1216e-04*BIRDS_NO + 31.7581e-02*CHANGELL_FT + 
                  24.813e-03*AirportWindSpInst_mph + 
                  19.7976e-02*SQUAREROOT[AirportRain48W_in]
    
    Target variable: ECOLI_LOG10 (already in log scale)
    Predictors: LOG10[TURB_NTRU], RHUM_PCT, WTEMP_CEL, BIRDS_NO, 
                CHANGELL_FT, AirportWindSpInst_mph, SQUAREROOT[AirportRain48W_in]
    """
    # the USGS model coefficients for comparison
    USGS_COEFFICIENTS = {
        'intercept': -1.01271,
        'LOG10_TURB_NTRU': 0.683547,
        'RHUM_PCT': 0.00793788,
        'WTEMP_CEL': 0.0525621,
        'BIRDS_NO': 0.00181216,
        'CHANGELL_FT': 0.317581,
        'AirportWindSpInst_mph': 0.024813,
        'SQRT_AirportRain48W_in': 0.197976
    }
    USGS_METRICS = {
        'r_squared': 0.47697,
        'adj_r_squared': 0.4747,
        'rmse': 0.4841,
        'sensitivity': 0.25581,
        'specificity': 0.97381,
        'accuracy': 0.90713
    }
    def __init__(self):
        """Initialise the Beach6 model."""
        self.model = LinearRegression()
        self.feature_names = [
            'LOG10_TURB_NTRU',
            'RHUM_PCT',
            'WTEMP_CEL',
            'BIRDS_NO',
            'CHANGELL_FT',
            'AirportWindSpInst_mph',
            'SQRT_AirportRain48W_in'
        ]
        self.is_fitted = False
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads and preprocess Beach6 data.
        Args:
            filepath: Path to the CSV data file 
        Returns:
            Preprocessed DataFrame
        """
        df = pd.read_csv(filepath)
        # parses date if needed
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates transformed features for the model.
        Args:
            df: Raw dataframe with original variables
        Returns:
            DataFrame with transformed features
        """
        df_features = df.copy()
        # target variable ECOLI_LOG10 is already in the data
        # creates LOG10 of turbidity + adds small constant to handle zeroes
        df_features['LOG10_TURB_NTRU'] = np.log10(df_features['TURB_NTRU'] + 0.1)
        # RHUM_PCT is already in the data
        # WTEMP_CEL is already in the data
        # BIRDS_NO is already in the data
        # CHANGELL_FT is already in the data
        # AirportWindSpInst_mph is already in the data
        # creates SQRT of rainfall
        df_features['SQRT_AirportRain48W_in'] = np.sqrt(df_features['AirportRain48W_in'])
        
        return df_features
    
    def fit(self, df: pd.DataFrame) -> 'Beach6EcoliModel':
        """
        Fits the linear regression model.
        Args:
            df: DataFrame with features (must already be transformed)  
        Returns:
            Self for method chaining
        """
        # prepare the features and target
        X = df[self.feature_names]
        y = df['ECOLI_LOG10']
        # fits the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new data.
        Args:
            df: DataFrame with features (must already be transformed)
        Returns:
            Array of predictions (LOG10 scale)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_names]
        return self.model.predict(X)
    
    def predict_concentration(self, df: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions and convert back to CFU/100mL scale.
        Args:
            df: DataFrame with features (must already be transformed)
        Returns:
            Array of predictions (CFU/100mL scale)
        """
        log_predictions = self.predict(df)
        return 10 ** log_predictions
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates model performance.
        Args:
            df: DataFrame with features and target (must already be transformed)
        Returns:
            Dictionary of evaluation metrics
        """
        X = df[self.feature_names]
        y_true = df['ECOLI_LOG10']
        y_pred = self.predict(df)
        
        n = len(y_true)
        p = len(self.feature_names)
        
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # calculates sensitivity, specificity, accuracy using threshold of:
        # LOG10(235) = 2.371 for E. coli standard
        threshold = np.log10(235)
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        true_negatives = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / n
        
        return {
            'r_squared': r2,
            'adj_r_squared': adj_r2,
            'rmse': rmse,
            'mae': mae,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'n_observations': n
        }
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Gets model coefficients.
        
        Returns:
            Dictionary of coefficient names and values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        coefs = {
            'intercept': self.model.intercept_
        }
        for name, coef in zip(self.feature_names, self.model.coef_):
            coefs[name] = coef
            
        return coefs
    
    def compare_with_usgs(self) -> pd.DataFrame:
        """
        Compares fitted coefficients with USGS model.
        
        Returns:
            DataFrame comparing coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        our_coefs = self.get_coefficients()
        
        comparison = []
        for key, usgs_val in self.USGS_COEFFICIENTS.items():
            our_val = our_coefs.get(key, np.nan)
            diff = our_val - usgs_val
            pct_diff = (diff / usgs_val * 100) if usgs_val != 0 else np.nan
            
            comparison.append({
                'coefficient': key,
                'usgs_value': usgs_val,
                'our_value': our_val,
                'difference': diff,
                'pct_difference': pct_diff
            })
        return pd.DataFrame(comparison)


def print_model_summary(model, model_name: str, metrics: Dict[str, float], df: pd.DataFrame):
    """
    Prints a comprehensive model summary.
    
    Args:
        model: Fitted model instance
        model_name: Name of the model for display
        metrics: Dictionary of evaluation metrics
        df: DataFrame used for training
    """
    print(f"{model_name} Model Summary")
    
    # model equation
    print("Model Equation:")
    coefs = model.get_coefficients()
    equation_parts = [f"{coefs['intercept']:.6e}"]
    for feat_name in model.feature_names:
        coef = coefs[feat_name]
        sign = '+' if coef >= 0 else '-'
        equation_parts.append(f"{sign} {abs(coef):.6e}*{feat_name}")
    
    print(f"LOG10[E.coli] = {' '.join(equation_parts)}")
    
    # coefficients table
    print(f"\n{'Coefficients:'}")
    print(f"{'Parameter':<30} {'Coefficient':>15}")
    print(f"{'Intercept':<30} {coefs['intercept']:>15.6e}")
    for feat_name in model.feature_names:
        print(f"{feat_name:<30} {coefs[feat_name]:>15.6e}")
    
    # eval metrics
    print(f"\n{'Evaluation Metrics:'}")
    print(f"{'Metric':<30} {'Value':>15}")
    print(f"{'N Observations':<30} {metrics['n_observations']:>15.0f}")
    print(f"{'R-squared':<30} {metrics['r_squared']:>15.5f}")
    print(f"{'Adjusted R-squared':<30} {metrics['adj_r_squared']:>15.5f}")
    print(f"{'RMSE':<30} {metrics['rmse']:>15.5f}")
    print(f"{'MAE':<30} {metrics['mae']:>15.5f}")
    print(f"{'Sensitivity':<30} {metrics['sensitivity']:>15.5f}")
    print(f"{'Specificity':<30} {metrics['specificity']:>15.5f}")
    print(f"{'Accuracy':<30} {metrics['accuracy']:>15.5f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    print("USGS E. coli Prediction Models")
    print("\nThis module provides classes for recreating USGS E. coli prediction models.")
    print("\nAvailable classes:")
    print("  - HuntingtonEcoliModel: For Huntington Beach, Pennsylvania")
    print("  - Beach6EcoliModel: For Beach6, Ohio")
    print("\nRefer to models.ipynb for usage examples.")