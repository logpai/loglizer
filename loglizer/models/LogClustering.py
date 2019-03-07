"""
The implementation of Log Clustering model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen. Log Clustering 
        based Problem Identification for Online Service Systems. International Conference
        on Software Engineering (ICSE), 2016.

"""

import numpy as np
import pprint
from scipy.special import expit
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from ..utils import metrics


class LogClustering(object):

    def __init__(self, max_dist=0.3, anomaly_threshold=0.3, mode='online', num_bootstrap_samples=1000):
        """
        Attributes
        ----------
            max_dist: float, the threshold to stop the clustering process
            anomaly_threshold: float, the threshold for anomaly detection
            mode: str, 'offline' or 'online' mode for clustering
            num_bootstrap_samples: int, online clustering starts with a bootstraping process, which
                determines the initial cluster representatives offline using a subset of samples 
            representatives: ndarray, the representative samples of clusters, of shape 
                num_clusters-by-num_events
            cluster_size_dict: dict, the size of each cluster, used to update representatives online 
        """
        self.max_dist = max_dist
        self.anomaly_threshold = anomaly_threshold
        self.mode = mode
        self.num_bootstrap_samples = num_bootstrap_samples
        self.representatives = list()
        self.cluster_size_dict = dict()

    def fit(self, X):   
        print('====== Model summary ======')         
        if self.mode == 'offline':
            # The offline mode can process about 10K samples only due to huge memory consumption.
            self._offline_clustering(X)
        elif self.mode == 'online':
            # Bootstrapping phase
            if self.num_bootstrap_samples > 0:
                X_bootstrap = X[0:self.num_bootstrap_samples, :]
                self._offline_clustering(X_bootstrap)
            # Online learning phase
            if X.shape[0] > self.num_bootstrap_samples:
                self._online_clustering(X)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            min_dist, min_index = self._get_min_cluster_dist(X[i, :])
            if min_dist > self.anomaly_threshold:
                y_pred[i] = 1
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n' \
              .format(precision, recall, f1))
        return precision, recall, f1

    def _offline_clustering(self, X):
        print('Starting offline clustering...')
        p_dist = pdist(X, metric=self._distance_metric)
        Z = linkage(p_dist, 'complete')
        cluster_index = fcluster(Z, self.max_dist, criterion='distance')
        self._extract_representatives(X, cluster_index)
        print('Processed {} instances.'.format(X.shape[0]))
        print('Found {} clusters offline.\n'.format(len(self.representatives)))
        # print('The representive vectors are:')
        # pprint.pprint(self.representatives.tolist())

    def _extract_representatives(self, X, cluster_index):
        num_clusters = len(set(cluster_index))
        for clu in range(num_clusters):
            clu_idx = np.argwhere(cluster_index == clu + 1)[:, 0]
            self.cluster_size_dict[clu] = clu_idx.shape[0]
            repre_center = np.average(X[clu_idx, :], axis=0)
            self.representatives.append(repre_center)

    def _online_clustering(self, X):
        print("Starting online clustering...")
        for i in range(self.num_bootstrap_samples, X.shape[0]):
            if (i + 1) % 2000 == 0:
                print('Processed {} instances.'.format(i + 1))
            instance_vec = X[i, :]
            if len(self.representatives) > 0:
                min_dist, clu_id = self._get_min_cluster_dist(instance_vec)
                if min_dist <= self.max_dist:
                    self.cluster_size_dict[clu_id] += 1
                    self.representatives[clu_id] = self.representatives[clu_id] \
                                                 + (instance_vec - self.representatives[clu_id]) \
                                                 / self.cluster_size_dict[clu_id]
                    continue
            self.cluster_size_dict[len(self.representatives)] = 1
            self.representatives.append(instance_vec)
        print('Processed {} instances.'.format(X.shape[0]))
        print('Found {} clusters online.\n'.format(len(self.representatives)))
        # print('The representive vectors are:')
        # pprint.pprint(self.representatives.tolist())

    def _distance_metric(self, x1, x2):
        norm= LA.norm(x1) * LA.norm(x2)
        distance = 1 - np.dot(x1, x2) / (norm + 1e-8)
        if distance < 1e-8:
            distance = 0
        return distance

    def _get_min_cluster_dist(self, instance_vec):
        min_index = -1
        min_dist = float('inf')
        for i in range(len(self.representatives)):
            cluster_rep = self.representatives[i]
            dist = self._distance_metric(instance_vec, cluster_rep)
            if dist < 1e-8:
                min_dist = 0
                min_index = i
                break
            elif dist < min_dist:
                min_dist = dist
                min_index = i
        return min_dist, min_index


