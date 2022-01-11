"""
The implementation of the LocalOutlierFactor for anomaly detection.

Authors:
    Hans Aschenloher

Reference:
    LOF: Identifying Density-Based Local Outliers, by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, Jörg Sander.
"""



import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
from ..utils import metrics

class LocalOutlinerFactor(LOF):
    def __init__(self, n_neighbors=10, algorithm='auto', contamination='auto', leaf_size=30, metric='minkowski', p=2):
        """
        Auguments
        ---------
        n_neighbors: int, default=20
        algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
        leaf_size:int, default=30
        metric: str or callable, default=’minkowski’
        p: int, default=2
        metric_params: dict, default=None. Additional keyword arguments for the metric function.
        contamination: ‘auto’ or float, default=’auto’
        novelty: bool, default=False. True if you want to use LocalOutlierFactor for novelty detection.
            In this case be aware that you should only use predict, decision_function and score_samples
            on new unseen data and not on the training set.
        n_jobs: int, default=None. The number of parallel jobs to run for neighbors search.
            None means 1 and -1 means all Processors

        Reference
        ---------
            For more information, please visit https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
	    """

        super(LocalOutlinerFactor, self).__init__(n_neighbors=n_neighbors, algorithm=algorithm, contamination=contamination, leaf_size=leaf_size, metric=metric, p=p)

    def fit(self, X):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('====== Model summary ======')
        super(LocalOutlinerFactor, self).fit(X)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """

        y_pred = super(LocalOutlinerFactor, self).predict(X)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

