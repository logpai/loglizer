"""
The utility functions of loglizer

Authors: 
    LogPAI Team

"""

from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def metrics(y_pred, y_true):
    """ Calucate evaluation metrics for precision, recall, and f1.

    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1

