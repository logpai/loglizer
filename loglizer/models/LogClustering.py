"""
The implementation of Log Clustering model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen. Log Clustering 
        based Problem Identification for Online Service Systems. International Conference
        on Software Engineering, 2016.

"""

import numpy as np
import pprint
from scipy.special import expit
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from ..utils import metrics

class LogClustering(object):

    def __init__(self, max_dist=0.3, anomaly_threshold=0.3):
        """
        Attributes
        ----------
            max_dist: float, the threshold to stop the clustering process
            anomaly_threshold: float, the threshold for anomaly detection
            representives: ndarray, the representative samples of clusters, 
                shape num_clusters-by-num_events

        """
        self.max_dist = max_dist
        self.anomaly_threshold = anomaly_threshold
        self.representives = None

    def fit(self, X, mode='offline'):   
        print('====== Model summary ======')         
        if mode == 'offline':
            self._offline_clustering(X)

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
        p_dist = pdist(X, metric=self._distance_metric)
        Z = linkage(p_dist, 'complete')
        cluster_index = fcluster(Z, self.max_dist, criterion='distance')
        representative_index = self._extract_representatives(p_dist, cluster_index)
        self.representives = X[representative_index, :]
        print('Found {} clusters in offline clustering'.format(len(self.representives)))
        print('The representive feature vectors are:')
        pprint.pprint(self.representives.tolist())
        print('')

    def _distance_metric(self, x1, x2):
        norm= LA.norm(x1) * LA.norm(x2)
        distance = 1 - np.dot(x1, x2) / (norm + 1e-8)
        if distance < 1e-8:
            distance = 0
        return distance

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
        return representative_index


def anomalyDetect(para, succ_index_list, fail_index_list, train_base_data, train_online_data, testing_data, train_online_label, testing_label, total_inst_num ):
    # clustering
    fail_cluster_results, fail_index_per_clu = clustering(para, fail_index_list, fail_data)
    print('failure data clustering finished...')
    succ_cluster_results, succ_index_per_clu = clustering(para, succ_index_list, succ_dta)
    print('success data clustering finished...')

    # extract representatives for each cluster of data
    dis_sum_list = np.zeros(total_inst_num)  # A one dimension list of all zero with size of totalLineNum
    fail_repre = extract_repre(train_base_data, fail_index_per_clu, dis_sum_list)
    succ_repre = extract_repre(train_base_data, succ_index_per_clu, dis_sum_list)

    # online learning
    train_base_size = train_base_data.shape[0]
    online_learn(para, train_online_data, train_online_label, dis_sum_list, fail_repre, succ_repre, fail_index_per_clu, succ_index_per_clu, train_base_data, train_base_size)

    # detect anomalies
    predict_label = detect(para, fail_repre, succ_repre, testing_data)



def online_learn(para, train_online_data, train_online_label, dis_sum_list, fail_repre, succ_repre, fail_index_per_clu, succ_index_per_clu, train_base_data, train_base_size):
    print("Start online learning...")
    train_online_size = train_online_data.shape[0]
    threshold = para['repre_threshold']
    for i in range(train_online_size):
        online_inst = train_online_data[i]
        if train_online_label[i] == 1:   # failure data cluster
            min_value, min_index = cal_dist(online_inst, fail_repre)
            if min_value <= threshold:
                cluster_index = min_index   # index of cluster label for new online instance
                update_repre(train_base_size + i, online_inst, fail_index_per_clu, train_online_data, fail_repre, dis_sum_list, train_base_data, cluster_index)
            else:
                fail_index_per_clu.append([train_base_size + i])
                fail_repre.append(online_inst)
        else:
            min_value, min_index = cal_dist(online_inst, succ_repre)
            if min_value <= threshold:
                cluster_index = min_index
                update_repre(train_base_size + i, online_inst, succ_index_per_clu, train_online_data, succ_repre, dis_sum_list, train_base_data, cluster_index)
            else:
                succ_index_per_clu.append([train_base_size + i])
                succ_repre.append(online_inst)


def update_repre(inst_index, online_inst, index_per_clu, train_online_data, represent, dis_sum_list, train_base_data, cluster_index):
    update_dis = []
    train_base_size = train_base_data.shape[0]
    index_in_each_cluster = index_per_clu[cluster_index]
    for ind in index_in_each_cluster:
        # online data
        if ind >= train_base_size:
            new_dist = compute_dist(online_inst, train_online_data[ind-train_base_size])
        else:
            new_dist = compute_dist(online_inst, train_base_data[ind])
        dis_sum_list[ind] += new_dist
        update_dis.append(new_dist)

    # add current log index into the current cluster
    index_per_clu[cluster_index].append(inst_index)

    # update newInstance data itself
    if dis_sum_list[inst_index] == 0:
        dis_sum_list[inst_index] += sum(update_dis)
    else:
        print('ERROR')

    #if this row is the same as the representive vector,then there is no need to find the new representive as they must be the same
    # choose the minimum value as the representive vector
    if not np.allclose(online_inst, represent[cluster_index]):
        part_dis_sum_list = dis_sum_list[index_in_each_cluster]
        min_index = index_in_each_cluster[np.argmin(part_dis_sum_list)]
        if min_index >= train_base_size:
            represent[cluster_index] = train_online_data[min_index - train_base_size]
        else:
            represent[cluster_index] = train_base_data[min_index]


def cal_dist(online_inst, represents):
    min_index = -1
    min_value = float('inf')
    for i, re in enumerate(represents):
        if np.allclose(online_inst, re):
            min_index = i
            min_value = 0
            break
        dis = compute_dist(online_inst, re)
        if dis < min_value:
            min_value = dis
            min_index = i
    return min_value, min_index






