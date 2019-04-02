"""
The implementation of PCA model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan. 
        Large-Scale System Problems Detection by Mining Console Logs. ACM 
        Symposium on Operating Systems Principles (SOSP), 2009.

"""

import numpy as np
from ..utils import metrics

class PCA(object):

    def __init__(self, n_components=0.95, threshold=None, c_alpha=3.2905):
        """ The PCA model for anomaly detection

        Attributes
        ----------
            proj_C: The projection matrix for projecting feature vector to abnormal space
            n_components: float/int, number of principal compnents or the variance ratio they cover
            threshold: float, the anomaly detection threshold. When setting to None, the threshold 
                is automatically caculated using Q-statistics
            c_alpha: float, the c_alpha parameter for caculating anomaly detection threshold using 
                Q-statistics. The following is lookup table for c_alpha:
                c_alpha = 1.7507; # alpha = 0.08
                c_alpha = 1.9600; # alpha = 0.05
                c_alpha = 2.5758; # alpha = 0.01
                c_alpha = 2.807; # alpha = 0.005
                c_alpha = 2.9677;  # alpha = 0.003
                c_alpha = 3.2905;  # alpha = 0.001
                c_alpha = 3.4808;  # alpha = 0.0005
                c_alpha = 3.8906;  # alpha = 0.0001
                c_alpha = 4.4172;  # alpha = 0.00001
        """

        self.proj_C = None
        self.components = None
        self.n_components = n_components
        self.threshold = threshold
        self.c_alpha = c_alpha


    def fit(self, X):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('====== Model summary ======')
        num_instances, num_events = X.shape
        X_cov = np.dot(X.T, X) / float(num_instances)
        U, sigma, V = np.linalg.svd(X_cov)
        n_components = self.n_components
        if n_components < 1:
            total_variance = np.sum(sigma)
            variance = 0
            for i in range(num_events):
                variance += sigma[i]
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1

        P = U[:, :n_components]
        I = np.identity(num_events, int)
        self.components = P
        self.proj_C = I - np.dot(P, P.T)
        print('n_components: {}'.format(n_components))
        print('Project matrix shape: {}-by-{}'.format(self.proj_C.shape[0], self.proj_C.shape[1]))

        if not self.threshold:
            # Calculate threshold using Q-statistic. Information can be found at:
            # http://conferences.sigcomm.org/sigcomm/2004/papers/p405-lakhina111.pdf
            phi = np.zeros(3)
            for i in range(3):
                for j in range(n_components, num_events):
                    phi[i] += np.power(sigma[j], i + 1)
            h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] * phi[1])
            self.threshold = phi[0] * np.power(self.c_alpha * np.sqrt(2 * phi[1] * h0 * h0) / phi[0]
                                               + 1.0 + phi[1] * h0 * (h0 - 1) / (phi[0] * phi[0]), 
                                               1.0 / h0)
        print('SPE threshold: {}\n'.format(self.threshold))

    def predict(self, X):
        assert self.proj_C is not None, 'PCA model needs to be trained before prediction.'
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_a = np.dot(self.proj_C, X[i, :])
            SPE = np.dot(y_a, y_a)
            if SPE > self.threshold:
                y_pred[i] = 1
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

