import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score


class Evaluation:
    def __init__(self, pipeline, X_test, y_test, config):
        self.config = config
        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.pipeline.predict(self.X_test)
        self.y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

    def evaluate_model(self):
        # Make predictions for test data
        if self.config['name'] == 'marketing_campaign':
            self.y_pred = np.argmax(self.y_pred, axis=1)

        print(confusion_matrix(self.y_test, self.y_pred))
        print(classification_report(self.y_test, self.y_pred))
        print(f"Accuracy: {accuracy_score(self.y_test, self.y_pred):.6f}")

    def get_auc(self):
        return roc_auc_score(self.y_test, self.y_pred_proba)
    
    def get_roc(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
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

    @staticmethod
    def get_eval1(clf, X, y):
        # Cross Validation to test and anticipate overfitting problem
        scores1 = cross_val_score(clf, X, y, cv=2, scoring='accuracy')
        scores2 = cross_val_score(clf, X, y, cv=2, scoring='precision')
        scores3 = cross_val_score(clf, X, y, cv=2, scoring='recall')
        scores4 = cross_val_score(clf, X, y, cv=2, scoring='roc_auc')

        # The mean score and standard deviation of the score estimate
        print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
        print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
        print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
        print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))

    @staticmethod
    def get_eval2(clf, X_train, y_train, X_test, y_test):
        # Cross Validation to test and anticipate overfitting problem
        scores1 = cross_val_score(clf, X_test, y_test, cv=2, scoring='accuracy')
        scores2 = cross_val_score(clf, X_test, y_test, cv=2, scoring='precision')
        scores3 = cross_val_score(clf, X_test, y_test, cv=2, scoring='recall')
        scores4 = cross_val_score(clf, X_test, y_test, cv=2, scoring='roc_auc')

        # The mean score and standard deviation of the score estimate
        print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
        print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
        print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
        print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))