import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.pipeline_manager import get_pipeline
from src.utils import get_roc  # Assuming get_roc is defined in utils.py
from xgboost import XGBClassifier

def run_pipeline(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    pipeline, X, y = get_pipeline(config)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Fit the preprocessor and transform the data
    X_train_processed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Extract the classifier and set the eval_set for early stopping
    model = pipeline.named_steps['classifier']
    eval_set = [(X_train_processed, y_train), (X_test_processed, y_test)]
    
    # Fit the model with early stopping
    model.fit(X_train_processed, y_train, eval_set=eval_set, eval_metric='auc', early_stopping_rounds=100, verbose=100)
    
    # Replace the classifier in the pipeline with the fitted model
    pipeline.steps[-1] = ('classifier', model)
    
    # Save the pipeline
    model_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), f"models/{config['name']}_pipeline.pkl"))
    joblib.dump(pipeline, model_filename)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    if config['name'] == 'marketing_campaign':
        y_pred = np.argmax(y_pred, axis=1)
    
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.6f}")

    get_roc(y_test, y_pred_proba)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python run_pipeline.py <config_path>")
    #     sys.exit(1)
    config_path = 'MLOps-24/configs/german_credit_config.json'
    # config_path = 'MLOps-24/configs/marketing_campaign_config.json'
    config_path = sys.argv[1]
    run_pipeline(config_path)
