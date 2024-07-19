import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
import shap
import numpy as np
import pandas as pd
import scipy.stats as kde
from xgboost import plot_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import shap
import pandas as pd
import numpy as np
import joblib
from src.pipeline_manager import get_pipeline
import os

def get_roc(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="upper left")
    plt.show()

# Plot Feature Importance
def plot_feature_importance(model):
    # plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=10, importance_type='gain', xlabel='Feature Importance (Gain)')
    plt.title('Top 10 Feature Importances')
    plt.show()

# Get the top features based on gain
def get_top_features(model, num_features=10):
    top_features = model.get_booster().get_score(importance_type='gain')
    top_features_sorted = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:num_features]
    top_features_names = [int(feature[1:]) for feature, _ in top_features_sorted]
    return top_features_names
    
def plot_shap_summary(config, model, X_test_processed, all_feature_names):
    explainer = shap.TreeExplainer(model)
    if config['name'] == 'marketing_campaign':
        shap_values = explainer.shap_values(X_test_processed, check_additivity=False)[1]
    else:
        shap_values = explainer.shap_values(X_test_processed)
    shap.summary_plot(shap_values, X_test_processed, feature_names=all_feature_names)


def select_features_based_on_shap(config, model, X_train, feature_names, threshold=0.01):
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    # Computes SHAP values for the training set
    explainer = shap.TreeExplainer(model)
    # shap_values = explainer(X_train_df)
    if config['name'] == 'marketing_campaign':
        shap_values = explainer.shap_values(X_train, check_additivity=False)[1]
    else:
        shap_values = explainer.shap_values(X_train)
    # Calculate the mean absolute SHAP values for each feature
    shap_summaries = np.abs(shap_values).mean(axis=0)
    
    # Identify features that meet the threshold
    selected_features = [X_train_df.columns[i] for i in range(len(shap_summaries)) if shap_summaries[i] > threshold]
    
    return selected_features

def plot_top_features_shap_dependence(config, model, top_features_names, X_test_processed, all_feature_names):
    explainer = shap.TreeExplainer(model)
    if config['name'] == 'marketing_campaign':
        shap_values = explainer.shap_values(X_test_processed, check_additivity=False)[1]
    else:
        shap_values = explainer.shap_values(X_test_processed)
        
    for feature in top_features_names:
        shap.dependence_plot(feature, shap_values, X_test_processed, feature_names=all_feature_names)
        
def plot_metric_drops(metric_results):
    # Plot metric drops if found
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
    plt.gca()#.invert_xaxis()
    plt.legend()
    plt.show()
    return vulnerable_features
    
        
def plot_partial_dependence(model, X_train_processed, all_feature_names, features_to_plot):
    features_indices = [list(all_feature_names).index(feature) for feature in features_to_plot]

    fig, ax = plt.subplots(1, len(features_to_plot), figsize=(10, 5), tight_layout=True)
    # Wrap ax in a list if there is only one subplot to ensure consistent indexing
    if len(features_to_plot) == 1:
        ax = [ax]
    for i, feature_index in enumerate(features_indices):
        PartialDependenceDisplay.from_estimator(model, X_train_processed, features=[feature_index], feature_names=all_feature_names, ax=ax[i])
        ax[i].set_title(f'{all_feature_names[feature_index]}')

    plt.suptitle('Partial Dependence Plots for Selected Features')
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.85)
    plt.show()

def uncertainty_baseline_ensemble_monte_carlo(model, X_train, y_train, X_test, n_models=10):
    predictions = np.zeros((X_test.shape[0], n_models))

    for i in range(n_models):
        idx = np.random.choice(np.arange(X_train.shape[0]), size=int(0.8 * X_train.shape[0]), replace=True)
        # print(X_train , y_train)
        X_train_bs, y_train_bs = X_train[idx], y_train[idx]
        
        model.fit(X_train_bs, y_train_bs)
        predictions[:, i] = model.predict_proba(X_test)[:, 1]

    uncertainty = predictions.std(axis=1)

    plt.scatter(predictions.mean(axis=1), uncertainty)
    plt.xlabel('Predicted Probability (Mean)')
    plt.ylabel('Uncertainty (Standard Deviation)')
    plt.title('Predicted Probability vs. Uncertainty')
    plt.show()
    
def plot_shap_dependence(config, model, X_test_processed, all_feature_names, features_to_plot):
    explainer = shap.TreeExplainer(model)
    if config['name'] == 'marketing_campaign':
        shap_values = explainer.shap_values(X_test_processed, check_additivity=False)[1]
    else:
        shap_values = explainer.shap_values(X_test_processed)

    for feature in features_to_plot:
        feature_index = list(all_feature_names).index(feature)
        shap.dependence_plot(feature_index, shap_values, X_test_processed, feature_names=all_feature_names)

# Analyze feature performance
def analyze_feature_performance(config, model, X_test_processed, y_test, all_feature_names, metric='auc'): # accuracy
    # Create a DataFrame for the test set including predictions and accuracy_bool
    out = pd.DataFrame(X_test_processed, columns=all_feature_names)
    out['target'] = y_test.values
    y_pred = model.predict(X_test_processed)
    if config['name'] == 'marketing_campaign':
        y_pred = np.argmax(y_pred, axis=1)
    out['accuracy_bool'] = (np.array(y_test).flatten() == np.array(y_pred))
    
    # Function to get HPD mask
    def hpd_grid(column, percent):
        density = kde.gaussian_kde(column)
        x = np.linspace(np.min(column), np.max(column), 2000)
        y = density.evaluate(x)
        threshold = np.percentile(y, 100 * (1 - percent))
        mask = density(column) > threshold
        return x, y, threshold, mask

    # Function to get accuracy
    def get_accuracy(config, mask, model):
        subset = out[mask]
        X_subset = subset.iloc[:, :-2]  # Exclude target and 'accuracy_bool' columns
        y_subset = subset['target']   # target column
        y_pred_subset = model.predict(X_subset)
        if config['name'] == 'marketing_campaign':
            y_pred_subset = np.argmax(y_pred_subset, axis=1)
        
        return accuracy_score(y_subset, y_pred_subset)
    
    # Function to get AUC
    def get_auc(config, mask, model):
        subset = out[mask]
        X_subset = subset.iloc[:, :-2]  # Exclude target and 'accuracy_bool' columns
        y_subset = subset['target']   # target column
        y_pred_subset = model.predict_proba(X_subset)[:, 1]

        return roc_auc_score(y_subset, y_pred_subset)
    
    # List to store accuracy or AUC results
    metric_results = []

    # Iterate over each feature in the dataset
    for feature in all_feature_names:
        print(f"Analyzing feature: {feature}")
        prior_metric = None
        feature_metric_results = []

        # Iterating over different HDR levels
        start, end, increment = 0.5, 0.95, 0.05
        percents = np.arange(start, end, increment)[::-1]

        for p in percents:
            # Run HPD and get mask of data in percent
            _, _, _, mask = hpd_grid(out[feature], percent=p)
            
            # Calculate accuracy or AUC
            if metric == 'accuracy':
                metric_value = get_accuracy(config, mask, model)
            elif metric == 'auc':
                metric_value = get_auc(config, mask, model)
            else:
                raise ValueError("Invalid metric. Please choose 'accuracy' or 'auc'.")
            
            # Save if metric decreases by > 0.001
            if prior_metric is not None and prior_metric - metric_value > 0.001:
                feature_metric_results.append((feature, p, metric_value, prior_metric - metric_value))
            
            # Reset prior variables
            prior_metric = metric_value

        metric_results.extend(feature_metric_results)
    
    return metric_results

def get_all_feature_names(config, preprocessor):
    # Get feature names after preprocessing
    numeric_features = config['numeric_features']
    categorical_features = config['categorical_features']
    
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    except AttributeError:
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    all_feature_names = np.concatenate([numeric_features, ohe_feature_names])
    
    return all_feature_names

def train_and_save_baseline_model(config):
    pipeline, X, y = get_pipeline(config)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Fit the preprocessor and transform the data
    preprocessor = pipeline.named_steps['preprocessor']
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Extract the classifier and set the eval_set for early stopping
    model = pipeline.named_steps['classifier']
    eval_set = [(X_train_processed, y_train), (X_test_processed, y_test)]
    
    # Fit the model with early stopping
    model.fit(X_train_processed, y_train, eval_set=eval_set, eval_metric='auc', early_stopping_rounds=100, verbose=100)
    
    # Retrieve the best number of boosting rounds
    best_n_estimators = model.best_iteration + 1

    # Update the model's n_estimators to the best number of boosting rounds
    model.set_params(**{'n_estimators': best_n_estimators})
    
    # Replace the classifier in the pipeline with the fitted model
    pipeline.steps[-1] = ('classifier', model)
    
    # Save the pipeline
    model_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_filename = os.path.join(model_directory, f"{config['name']}_improved_pipeline.pkl")
    joblib.dump(pipeline, model_filename)
    
    all_feature_names = get_all_feature_names(config, preprocessor)
    # print("All feature names:", len(all_feature_names), all_feature_names)
    
    return pipeline, model, all_feature_names, X_train_processed, X_test_processed, X_test, y_train, y_test