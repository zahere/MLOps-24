import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataLoaderPreprocessor:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        if self.config['name'] == 'german_credit':
            data = pd.read_csv(self.config['data_path'], names=self.config['names'], delimiter=self.config['sep'])
            data[self.config['target']].replace([1, 2], [1, 0], inplace=True)
        elif self.config['name'] == 'marketing_campaign':
            data = pd.read_csv(self.config['data_path'], delimiter=self.config['sep'])
            data[self.config['target']].replace(['yes', 'no'], [1, 0], inplace=True)
        else:
            raise Exception("Sorry, Unknown Configurations")
        
        
        X = data.drop(columns=[self.config['target']])
        y = data[self.config['target']]
        
        return X, y

    def get_preprocessor(self):
        numeric_features = self.config['numeric_features']
        categorical_features = self.config['categorical_features']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num','passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        return preprocessor
  
