from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from pandas.api.types import is_numeric_dtype

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import shap

class ModelEvaluator():
    def __init__(self, model, data, actual_name, expected_name, compare_name = None):
        self.model = model
        self.data: pd.DataFrame = data
        self.actual_name: str = actual_name
        self.expected_name: str = expected_name
        self.compare_name: str = compare_name
        
        self.actual = self.data[self.actual_name]
        self.expected = self.data[self.expected_name]
        if self.compare_name is not None:
            self.compare = self.data[self.compare_name]
    
class XGBModelEvaluator(ModelEvaluator):
    def __init__(self, model, data, actual_name, expected_name, compare_name = None):
        super().__init__(model, data, actual_name, expected_name, compare_name)
    
        self.feature_names = list(self.model.feature_names_in_)
        self.shap_values = None
    
    def plot_feature_importance(self, importance_type = "total_gain"):
        """Plot feature importance for the model"""
        xgb.plot_importance(self.model, importance_type = importance_type)
         
    def _get_shap_values(self):
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer(self.data[self.feature_names])

    def plot_shap_summary_plot(self, max_display=10):
        """Plot SHAP values for tree-based and other models"""
        if not(self.shap_values):
            self._get_shap_values()
        shap.summary_plot(self.shap_values, self.data[self.feature_names], max_display = max_display)
        
    def plot_pdp(self, feature_list):
        """Plot partial dependence plot for a given feature"""
        PartialDependenceDisplay.from_estimator(self.model, self.data[self.feature_names], feature_list)
        
    def plot_ice(self, feature_list):
        """Plot individual conditional expectation (ICE) plot for a given feature"""
        PartialDependenceDisplay.from_estimator(self.model, self.data[self.feature_names], feature_list, kind="both")

    def plot_ave(self):
        """Plot actual vs. predicted values"""
                
        # Plot actual vs. predicted values
        plt.scatter(self.actual, self.expected)
        plt.plot([0, max(self.actual)], [0, max(self.actual)], 'r--')
        plt.xlabel("Actual values")
        plt.ylabel("Predicted values")
        plt.show()
        
    def _get_feature_plot_data(self, feature):
            
        plot_dict = {
        'actual':self.actual,
        'expected':self.expected,
        'feature':self.data[feature]
        }
        plot_data = pd.DataFrame(plot_dict)
        plot_data.head()

        if is_numeric_dtype(plot_data['feature']) & (len(np.unique(plot_data['feature'])) > 50):
            bins = 10
            edges = np.linspace(plot_data['feature'].min(), plot_data['feature'].max(), bins+1).astype(int)
            labels = [f'({edges[i]}, {edges[i+1]}]' for i in range(bins)]
            plot_data['feature'] = pd.cut(plot_data['feature'], bins = bins, labels = labels)
            
        feature_plot_data = plot_data.groupby('feature').agg(
            actual = ('actual', 'mean'),
            expected = ('expected', 'mean'),
            exposure = ('actual', 'size'),
            ).reset_index()
        
        return feature_plot_data
    
    def plot_feature_ave(self, feature):
        
        feature_plot_data = self._get_feature_plot_data(feature)
    
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.bar(feature_plot_data['feature'],feature_plot_data['exposure'])
        ax2.plot(feature_plot_data['feature'], feature_plot_data['actual'])
        ax2.plot(feature_plot_data['feature'], feature_plot_data['expected'])

        ax1.set_xlabel("feature")
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            
        ax1.set_ylabel("Number of Games", fontsize=14)

        ax2.set_ylabel("Total Points Scored", fontsize=14)

        fig.suptitle("Testing", fontsize=20)
        fig.show()
    
class XGBClassifierEvaluator(XGBModelEvaluator):
    def __init__(self) -> None:
        super().__init__()
    
    def get_confusion_matrix(self, y_true, y_pred):
        """Return the confusion matrix for binary classification"""
        return confusion_matrix(y_true, y_pred)

    def get_roc_curve(self, y_true, y_pred):
        """Return the ROC curve for binary classification"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return fpr, tpr, thresholds

    def get_auc_score(self, y_true, y_pred):
        """Return the AUC score for binary classification"""
        return roc_auc_score(y_true, y_pred)
    
class XGBRegressorEvaluator(XGBModelEvaluator):

    def get_mae(self):
        """Return the mean absolute error for regression"""
        return mean_absolute_error(self.actual, self.expected)
    
    def get_mse(self):
        """Return the mean squared error for regression"""
        return mean_squared_error(self.actual, self.expected)
    
    def get_r2_score(self):
        """Return the R-squared score for regression"""
        return r2_score(self.actual, self.expected)
    
