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

    def __init__(self, max_dist=0.3, anomaly_threshold=0.3, mode='online', num_bootstrap_samples=10000):
        """
        Attributes
        ----------
            max_dist: float, the threshold to stop the clustering process
            anomaly_threshold: float, the threshold for anomaly detection
            model: str, 'offline' or 'online' mode for clustering
            num_bootstrap_samples: int, online clustering starts with a bootstraping process, which
                determines the initial cluster representives offline using a subset of samples 
            representives: ndarray, the representative samples of clusters, of shape 
                num_clusters-by-num_events
        """
        self.max_dist = max_dist
        self.anomaly_threshold = anomaly_threshold
        self.mode = mode
        self.num_bootstrap_samples = num_bootstrap_samples
        self.representives = None
        self.cluster_index = list()
        self.dist_sum_dict = dict()

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
            row = X[i, :]
            dist_list = []
            for j in range(self.representives.shape[0]):
                cluster_rep = self.representives[j, :]
                dist_list.append(self._distance_metric(cluster_rep, row))
            if min(dist_list) > self.anomaly_threshold:
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
        self.cluster_index = cluster_index.tolist()
        print('type(cluster_index', type(self.cluster_index))
        representative_index = self._extract_representatives(p_dist, cluster_index)
        self.representives = X[representative_index, :]
        print('Found {} clusters in offline clustering'.format(self.representives.shape[0]))
        # print('The representive feature vectors are:')
        # pprint.pprint(self.representives.tolist())

    def _extract_representatives(self, p_dist, cluster_index):
        representative_index = []
        dist_matrix = squareform(p_dist)
        num_clusters = len(set(cluster_index))
        for clu in range(num_clusters):
            clu_idx = np.argwhere(cluster_index == clu + 1)[:, 0]
            sub_dist_matrix = dist_matrix[clu_idx, :]
            sub_dist_matrix = sub_dist_matrix[:, clu_idx]
            dist_sum_vec = np.sum(sub_dist_matrix, axis=0)
            min_idx = np.argmin(dist_sum_vec)
            representative_index.append(clu_idx[min_idx])
            self.dist_sum_dict[clu] = dist_sum_vec.tolist()
        return representative_index

    def _online_clustering(self, X):
        print("Starting online clustering...")
        for i in range(self.num_bootstrap_samples, X.shape[0]):
            if (i + 1) % 1000 == 0:
                print('Processed {} instances.'.format(i + 1))
            instance_vec = X[i, :]
            if self.representives is None:
                clu_id = 0 # the first cluster
                self.cluster_index.append(clu_id + 1)
                self.dist_sum_dict[clu_id] = [0] # single instance
                self.representives = instance_vec.reshape((1, -1))
            else:
                min_dist, clu_id = self._get_min_cluster_dist(instance_vec)
                if min_dist <= self.max_dist:
                    instance_idx = np.argwhere(np.array(self.cluster_index) == clu_id + 1)[:, 0]
                    X_c = X[instance_idx, :]
                    self._update_representatives(X_c, instance_vec, clu_id)
                    self.cluster_index.append(clu_id + 1)
                else:
                    clu_id = self.representives.shape[0]
                    self.cluster_index.append(clu_id + 1)
                    self.dist_sum_dict[clu_id] = [0] # single instance
                    self.representives = np.vstack([self.representives, instance_vec])

        print('Found {} clusters in online clustering'.format(self.representives.shape[0]))
        # print('The representive feature vectors are:')
        # pprint.pprint(self.representives.tolist())

    def _update_representatives(self, X_c, instance_vec, clu_id):
        dist_sum = 0 
        dist_sum_list = self.dist_sum_dict[clu_id]
        for i in range(X_c.shape[0]):
            X_i = X_c[i, :]
            dist = self._distance_metric(X_i, instance_vec)
            dist_sum += dist
            dist_sum_list[i] += dist
        dist_sum_list.append(dist_sum)
        self.dist_sum_dict[clu_id] = dist_sum_list

        # only update representative when the new instance is different from the original
        if self._distance_metric(instance_vec, self.representives[clu_id]) > 0:
            # choose the minimum as the representive vector
            min_idx = np.argmin(dist_sum_list)
            if min_idx == len(dist_sum_list) - 1:
                self.representives[clu_id, :] = instance_vec
            else:
                self.representives[clu_id, :] = X_c[min_idx, :]

    def _distance_metric(self, x1, x2):
        norm= LA.norm(x1) * LA.norm(x2)
        distance = 1 - np.dot(x1, x2) / (norm + 1e-8)
        if distance < 1e-8:
            distance = 0
        return distance

    def _get_min_cluster_dist(self, instance_vec):
        min_index = -1
        min_dist = float('inf')
        for i in range(self.representives.shape[0]):
            cluster_rep = self.representives[i, :]
            dist = self._distance_metric(instance_vec, cluster_rep)
            if dist < 1e-8:
                min_dist = 0
                min_index = i
                break
            elif dist < min_dist:
                min_dist = dist
                min_index = i
        return min_dist, min_index


