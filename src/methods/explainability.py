import pandas as pd
import numpy as np
import shap
from sklearn.inspection import PartialDependenceDisplay
from xgboost import plot_importance
import matplotlib.pyplot as plt

class Explainability:
    def __init__(self, config, model, X_train, all_feature_names):
        self.config = config
        self.model = model
        self.X_train = X_train
        self.all_feature_names = all_feature_names
        self.explainer = shap.TreeExplainer(self.model)
        if self.config['name'] == 'marketing_campaign':
            self.shap_values = self.explainer.shap_values(self.X_train).mean(axis=2)
        else:
            self.shap_values = self.explainer.shap_values(self.X_train)
    
    # Plot Feature Importance (Gain)
    def plot_feature_importance(self):
        plot_importance(self.model, max_num_features=10, importance_type='gain', xlabel='Feature Importance (Gain)')
        plt.title('Top 10 Feature Importances (Gain)')
        plt.show()
            
    def plot_shap_summary(self):
        shap.summary_plot(self.shap_values, self.X_train, feature_names=self.all_feature_names)
    
    def select_features_based_on_shap(self, threshold=0.01):
        shap_summaries = np.abs(self.shap_values).mean(axis=0)
        selected_features = [self.all_feature_names[i] for i in range(len(shap_summaries)) if shap_summaries[i] > threshold]
        return selected_features
    
    def plot_top_features_shap_dependence(self, top_features_names):
        for feature in top_features_names:
            shap.dependence_plot(feature, self.shap_values, self.X_train, feature_names=self.all_feature_names)
    
    def plot_shap_dependence(self, features_to_plot):
        for feature in features_to_plot:
            feature_index = list(self.all_feature_names).index(feature)
            shap.dependence_plot(feature_index, self.shap_values, self.X_train, feature_names=self.all_feature_names)
    
    def plot_partial_dependence(self, features_to_plot):
        features_indices = [list(self.all_feature_names).index(feature) for feature in features_to_plot]
        fig, ax = plt.subplots(1, len(features_to_plot), figsize=(10, 5), tight_layout=True)
        if len(features_to_plot) == 1:
            ax = [ax]
        for i, feature_index in enumerate(features_indices):
            PartialDependenceDisplay.from_estimator(self.model, self.X_train, features=[feature_index], feature_names=self.all_feature_names, ax=ax[i])
            ax[i].set_title(f'{self.all_feature_names[feature_index]}')
        plt.suptitle('Partial Dependence Plots for Selected Features')
        plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.85)
        plt.show()
        
    def plot_shap_interaction(self, feature_pair):
        feature1, feature2 = feature_pair
        feature1_index = list(self.all_feature_names).index(feature1)
        feature2_index = list(self.all_feature_names).index(feature2)
        shap.dependence_plot(feature1_index, self.shap_values, self.X_train, feature_names=self.all_feature_names, interaction_index=feature2_index)