"""
The implementation of IsolationForest model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. Isolation Forest. International
        Conference on Data Mining (ICDM), 2008.

"""





import numpy as np
from sklearn.ensemble import IsolationForest as iForest
from ..utils import metrics

class IsolationForest(iForest):

    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.03, **kwargs):
        """ The IsolationForest model for anomaly detection

        Arguments
        ---------
            n_estimators : int, optional (default=100). The number of base estimators in the ensemble.
            max_samples : int or float, optional (default="auto")
                The number of samples to draw from X to train each base estimator.
                    - If int, then draw max_samples samples.
                    - If float, then draw max_samples * X.shape[0] samples.
                    - If "auto", then max_samples=min(256, n_samples).
                If max_samples is larger than the number of samples provided, all samples will be used 
                for all trees (no sampling).
            contamination : float in (0., 0.5), optional (default='auto')
                The amount of contamination of the data set, i.e. the proportion of outliers in the data 
                set. Used when fitting to define the threshold on the decision function. If 'auto', the 
                decision function threshold is determined as in the original paper.
            max_features : int or float, optional (default=1.0)
                The number of features to draw from X to train each base estimator.
                    - If int, then draw max_features features.
                    - If float, then draw max_features * X.shape[1] features.
            bootstrap : boolean, optional (default=False)
                If True, individual trees are fit on random subsets of the training data sampled with replacement. 
                If False, sampling without replacement is performed.
            n_jobs : int or None, optional (default=None)
                The number of jobs to run in parallel for both fit and predict. None means 1 unless in a 
                joblib.parallel_backend context. -1 means using all processors. 
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator; 
                If RandomState instance, random_state is the random number generator; 
                If None, the random number generator is the RandomState instance used by np.random.
        
        Reference
        ---------
            For more information, please visit https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        """

        super(IsolationForest, self).__init__(n_estimators=n_estimators, max_samples=max_samples, 
            contamination=contamination, **kwargs)


    def fit(self, X):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('====== Model summary ======')
        super(IsolationForest, self).fit(X)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        
        y_pred = super(IsolationForest, self).predict(X)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

