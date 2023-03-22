# -*- coding: utf-8 -*-
"""
The implementation of the logistic regression model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Peter Bodík, Moises Goldszmidt, Armando Fox, Hans Andersen. Fingerprinting 
        the Datacenter: Automated Classification of Performance Crises. The European 
        Conference on Computer Systems (EuroSys), 2010.

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from ..utils import metrics

class LR(object):

    def __init__(self, penalty='l2', C=100, tol=0.01, class_weight=None, max_iter=100):
        """ The Invariants Mining model for anomaly detection

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection
        """
        self.classifier = LogisticRegression(penalty=penalty, C=C, tol=tol, class_weight=class_weight,
                                             max_iter=max_iter)

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

    def predict_proba(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        y_pred = self.classifier.predict_proba(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
