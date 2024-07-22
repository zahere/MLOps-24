import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.data.data_loader_preprocessor import DataLoaderPreprocessor
from sklearn.model_selection import train_test_split
import os
import joblib
from imblearn.over_sampling import SMOTE

class PipelineManager:
    def __init__(self, config):
        self.config = config
        self.data_loader_preprocessor = DataLoaderPreprocessor(config)
        self.pipeline = None
        self.model = None
        self.all_feature_names = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_pipeline(self):
        X, y = self.data_loader_preprocessor.load_data()

        preprocessor = self.data_loader_preprocessor.get_preprocessor()
        params = self.config['xgboost_params']
        model = XGBClassifier(**params, eval_metric = 'auc', early_stopping_rounds=100,)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        return self.pipeline, X, y

    def get_all_feature_names(self, preprocessor):
        numeric_features = self.config['numeric_features']
        categorical_features = self.config['categorical_features']

        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        except AttributeError:
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

        self.all_feature_names = np.concatenate([numeric_features, ohe_feature_names])

        return self.all_feature_names

    def train_baseline_model(self):
        self.pipeline, X, y = self.get_pipeline()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        preprocessor = self.pipeline.named_steps['preprocessor']
        self.X_train_processed = preprocessor.fit_transform(X_train)
        self.X_test_processed = preprocessor.transform(X_test)

        model = self.pipeline.named_steps['classifier']
        eval_set = [(self.X_train_processed, y_train), (self.X_test_processed, y_test)]

        model.fit(self.X_train_processed, y_train, eval_set=eval_set,  verbose=100)

        best_n_estimators = model.best_iteration + 1
        model.set_params( **{'n_estimators': best_n_estimators}) 

        self.pipeline.steps[-1] = ('classifier', model)

        model_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        model_filename = os.path.join(model_directory, f"{self.config['name']}_improved_pipeline.pkl")
        joblib.dump(self.pipeline, model_filename)

        self.all_feature_names = self.get_all_feature_names(preprocessor)

        self.model = model
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return self.pipeline, self.model, self.all_feature_names, self.X_train_processed, self.X_test_processed, self.X_test, self.y_train, self.y_test
    
    
    def add_noise(self, X, numeric_features, noise_level=0.01):
        noise = np.random.normal(0, noise_level, X[numeric_features].shape)
        X_noisy = X.copy()
        X_noisy[numeric_features] += noise
        return X_noisy

    def train_model_with_data_augmentation(self):
        self.pipeline, X, y = self.get_pipeline()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Data Augmentation
        numeric_features = self.config['numeric_features']
        X_train_noisy = self.add_noise(X_train, numeric_features)
        y_train_noisy = y_train.copy()
        X_train_augmented = pd.concat([X_train, X_train_noisy])
        y_train_augmented = pd.concat([y_train, y_train_noisy])

        preprocessor = self.pipeline.named_steps['preprocessor']
        self.X_train_processed = preprocessor.fit_transform(X_train_augmented)
        self.X_test_processed = preprocessor.transform(X_test)

        model = self.pipeline.named_steps['classifier']
        eval_set = [(self.X_train_processed, y_train_augmented), (self.X_test_processed, y_test)]

        model.fit(self.X_train_processed, y_train_augmented, eval_set=eval_set, verbose=100)

        best_n_estimators = model.best_iteration + 1
        model.set_params(**{'n_estimators': best_n_estimators})

        self.pipeline.steps[-1] = ('classifier', model)

        model_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        model_filename = os.path.join(model_directory, f"{self.config['name']}_improved_pipeline.pkl")
        joblib.dump(self.pipeline, model_filename)

        self.all_feature_names = self.get_all_feature_names(preprocessor)

        self.model = model
        self.X_test = X_test
        self.y_train = y_train_augmented
        self.y_test = y_test

        return self.pipeline, self.model, self.all_feature_names, self.X_train_processed, self.X_test_processed, self.X_test, self.y_train, self.y_test

    
    def create_adversarial_examples(self, model, X, y, epsilon=0.01):
        dX = np.gradient(model.predict_proba(X)[:, 1])
        X_adv = X + epsilon * np.sign(dX.reshape(-1, 1) )
        return X_adv, y
    
    
    def train_model_with_adversarial_data(self):
        self.pipeline, X, y = self.get_pipeline()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        preprocessor = self.pipeline.named_steps['preprocessor']
        self.X_train_processed = preprocessor.fit_transform(X_train)
        self.X_test_processed = preprocessor.transform(X_test)

        model = self.pipeline.named_steps['classifier']
        eval_set = [(self.X_train_processed, y_train), (self.X_test_processed, y_test)]

        # Train initial model
        model.fit(self.X_train_processed, y_train, eval_set=eval_set, verbose=100)
        
        # Generate adversarial examples
        X_adv, y_adv = self.create_adversarial_examples(model, self.X_train_processed, y_train)
        X_train_augmented = np.concatenate([self.X_train_processed, X_adv])
        y_train_augmented =  np.concatenate([y_train, y_adv])

        # Retrain model with adversarial examples
        model.fit(X_train_augmented, y_train_augmented, eval_set=eval_set, verbose=100)

        best_n_estimators = model.best_iteration + 1
        model.set_params(**{'n_estimators': best_n_estimators})

        self.pipeline.steps[-1] = ('classifier', model)

        model_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        model_filename = os.path.join(model_directory, f"{self.config['name']}_improved_pipeline.pkl")
        joblib.dump(self.pipeline, model_filename)

        self.all_feature_names = self.get_all_feature_names(preprocessor)

        self.model = model
        self.X_test = X_test
        self.y_train = y_train_augmented
        self.y_test = y_test

        return self.pipeline, self.model, self.all_feature_names, self.X_train_processed, self.X_test_processed, self.X_test, self.y_train, self.y_test
