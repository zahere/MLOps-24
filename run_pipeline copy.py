import os
import sys
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
import matplotlib.pyplot as plt


class DataLoaderPreprocessor:
    def __init__(self, config):
        self.config = config

    def load_and_preprocess_data(self):
        # Load raw data
        data = pd.read_csv(self.config['data_path'], names=self.config['names'], delimiter=self.config['sep'])
        
        # Binarize the target column
        data[self.config['target']].replace([1, 2], [1, 0], inplace=True)
        
        # Separate features and target
        X = data.drop(columns=[self.config['target']])
        y = data[self.config['target']]
        
        # Define numerical and categorical features
        numeric_features = self.config['numeric_features']
        categorical_features = self.config['categorical_features']
        
        # Create a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        # Convert the processed data back to a DataFrame for easier debugging
        X_processed_df = pd.DataFrame(X_processed, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
        
        return X_processed_df, y

def get_roc(y_test, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plot of a ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="upper left")
    plt.show()

def run_pipeline(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Initialize the DataLoaderPreprocessor
    data_loader_preprocessor = DataLoaderPreprocessor(config)
    
    # Load and preprocess the data
    X_processed, y = data_loader_preprocessor.load_and_preprocess_data()
    
    # Print the processed data columns and types
    print("Processed Data Columns after loading:", X_processed.columns.tolist())
    print("Data types of X:")
    print(X_processed.dtypes)
    print("Data type of y:")
    print(y.dtypes)
    
    # Ensure all feature columns are converted to floats
    X_processed = X_processed.astype(float)
    y = y.astype(int)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=1)

    # Print shapes of the splits
    print("Shapes of the splits:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
    # Define XGBoost parameters
    params = {
        'n_estimators': 3000,
        'objective': 'binary:logistic',
        'learning_rate': 0.005,
        'subsample': 0.555,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'max_depth': 8,
        'n_jobs': -1
    }
    
    # Define and fit the model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc', early_stopping_rounds=100, verbose=100)
    
    # Save the model
    model_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), f"models/{config['name']}_pipeline.pkl"))
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(model, model_filename)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.6f}")

    # Plot ROC curve
    get_roc(y_test, y_pred_proba)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_pipeline(config_path)
