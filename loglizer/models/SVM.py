"""
The implementation of the SVM model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Yinglung Liang, Yanyong Zhang, Hui Xiong, Ramendra Sahoo. Failure Prediction 
        in IBM BlueGene/L Event Logs. IEEE International Conference on Data Mining
        (ICDM), 2007.

"""

import numpy as np
from sklearn import svm
from ..utils import metrics

class SVM(object):

    def __init__(self, penalty='l1', tol=0.1, C=1, dual=False, class_weight=None, 
                 max_iter=100):
        """ The Invariants Mining model for anomaly detection
        Arguments
        ---------
        See SVM API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        
        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = svm.LinearSVC(penalty=penalty, tol=tol, C=C, dual=dual, 
                                        class_weight=class_weight, max_iter=max_iter)

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
        
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
