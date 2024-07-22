import matplotlib.pyplot as plt
import numpy as np
# from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
class Uncertainty:
    def __init__(self, model, X_train, y_train, X_test, n_models=10):
        self.model = XGBClassifier(eval_metric = 'auc')
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.n_models = n_models
        self.predictions = np.zeros((X_test.shape[0], n_models))
    
    def baseline_ensemble_monte_carlo(self):
        for i in range(self.n_models):
            idx = np.random.choice(np.arange(self.X_train.shape[0]), size=int(0.8 * self.X_train.shape[0]), replace=True)
            X_train_bs, y_train_bs = self.X_train[idx], self.y_train[idx]
            # X_val_bs, y_val_bs = self.X_train[idx], self.y_train[idx]
            # X_train_bs, X_val_bs, y_train_bs, y_val_bs = train_test_split(self.X_train, self.y_train, test_size=0.8, random_state=1, shuffle=True)
            
            self.model.fit(X_train_bs, y_train_bs)
            self.predictions[:, i] = self.model.predict_proba(self.X_test)[:, 1]
        uncertainty = self.predictions.std(axis=1)
        plt.scatter(self.predictions.mean(axis=1), uncertainty)
        plt.xlabel('Predicted Probability (Mean)')
        plt.ylabel('Uncertainty (Standard Deviation)')
        plt.title('Predicted Probability vs. Uncertainty')
        plt.show()