"""
The implementation of the SVM model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Yinglung Liang, Yanyong Zhang, Hui Xiong, Ramendra Sahoo. Failure Prediction 
        in IBM BlueGene/L Event Logs. ICDM, 2007.

"""

import numpy as np
from sklearn import svm
from ..utils import metrics

class SVM(object):

    def __init__(self):
        """ The Invariants Mining model for anomaly detection

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = svm.LinearSVC(penalty='l1', tol=0.0001, C=1, dual=False, fit_intercept=True, intercept_scaling=1, class_weight='balanced', max_iter=1000)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        
        y_pred = list(self.classifier.predict(X))
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
