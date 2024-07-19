import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import scipy.stats as kde
import matplotlib.pyplot as plt

class FeaturePerformanceWeaknessAnalyzer:
    def __init__(self, config, model, X_test_processed, y_test, all_feature_names):
        self.config = config
        self.model = model
        self.X_test_processed = X_test_processed
        self.y_test = y_test
        self.all_feature_names = all_feature_names
        self.out = pd.DataFrame(X_test_processed, columns=all_feature_names)
        self.out['target'] = y_test.values
        self.y_pred = model.predict(X_test_processed)
        if config['name'] == 'marketing_campaign':
            self.y_pred = np.argmax(self.y_pred, axis=1)
        self.out['accuracy_bool'] = (np.array(y_test).flatten() == np.array(self.y_pred))
    
    def hpd_grid(self, column, percent):
        density = kde.gaussian_kde(column)
        x = np.linspace(np.min(column), np.max(column), 2000)
        y = density.evaluate(x)
        threshold = np.percentile(y, 100 * (1 - percent))
        mask = density(column) > threshold
        return x, y, threshold, mask
    
    def get_accuracy(self, mask):
        subset = self.out[mask]
        X_subset = subset.iloc[:, :-2]
        y_subset = subset['target']
        y_pred_subset = self.model.predict(X_subset)
        if self.config['name'] == 'marketing_campaign':
            y_pred_subset = np.argmax(y_pred_subset, axis=1)
        return accuracy_score(y_subset, y_pred_subset)
    
    def get_auc(self, mask):
        subset = self.out[mask]
        X_subset = subset.iloc[:, :-2]
        y_subset = subset['target']
        y_pred_subset = self.model.predict_proba(X_subset)[:, 1]
        return roc_auc_score(y_subset, y_pred_subset)
    
    def analyze_feature_performance(self, metric='auc'):
        metric_results = []
        
        for feature in self.all_feature_names:
            print(f"Analyzing feature: {feature}")
            prior_metric = None
            feature_metric_results = []
            start, end, increment = 0.5, 0.95, 0.05
            percents = np.arange(start, end, increment)[::-1]
            
            for p in percents:
                _, _, _, mask = self.hpd_grid(self.out[feature], percent=p)
                if metric == 'accuracy':
                    metric_value = self.get_accuracy(mask)
                elif metric == 'auc':
                    metric_value = self.get_auc(mask)
                else:
                    raise ValueError("Invalid metric. Please choose 'accuracy' or 'auc'.")
                if prior_metric is not None and prior_metric - metric_value > 0.001:
                    feature_metric_results.append((feature, p, metric_value, prior_metric - metric_value))
                prior_metric = metric_value
            
            metric_results.extend(feature_metric_results)
        
        return metric_results
    
    def plot_metric_drops(self, metric_results):
        vulnerable_features = []
        hdr_levels, scores, score_drops, features = zip(*[(r[1], r[2], r[3], r[0]) for r in metric_results])
        plt.figure(figsize=(10, 6))
        for feature in set(features):
            vulnerable_features.append(feature)
            feature_indices = [i for i, f in enumerate(features) if f == feature]
            plt.plot(np.array(hdr_levels)[feature_indices], np.array(score_drops)[feature_indices], marker='o', label=f'Feature: {feature}')
        plt.title('Metric Drops Across Different HDR Levels by Feature')
        plt.xlabel('HDR Level')
        plt.ylabel('Metric Drop')
        plt.gca()
        plt.legend()
        plt.show()
        return vulnerable_features
    
