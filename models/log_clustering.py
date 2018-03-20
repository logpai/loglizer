#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Shilin He'

import numpy as np
import math
from scipy.special import expit
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
import utils.evaluation as ev


def weighting(raw_data):
	# to avoid the case that a term never occurs in a document, we add 1 to the cnt
	inst_num, event_num = raw_data.shape
	weighted_matrix = np.zeros((inst_num, event_num), float)
	for j in range(event_num):
		cnt = np.count_nonzero(raw_data[:, j])
		for i in range(inst_num):
			weight = 0.5 * expit(math.log((inst_num+1)/float(cnt+1)))
			weighted_matrix[i, j] = raw_data[i, j] * weight
	print('weighted data size is', weighted_matrix.shape)
	total_inst_num = weighted_matrix.shape[0]
	return weighted_matrix, total_inst_num


def split_data(para, weighted_matrix, labels):
	inst_num = weighted_matrix.shape[0]
	print('seperating the initial training Data...')
	train_base_size = int(math.floor(inst_num * para['train_base_per']))
	train_base_online_size = int(math.floor(inst_num * (para['train_base_per'] + para['train_online_per'])))

	train_base_data = weighted_matrix[:train_base_size, :]
	train_online_data = weighted_matrix[train_base_size:train_base_online_size, :]
	testing_data = weighted_matrix[train_base_online_size:, :]

	labels = np.squeeze(labels).tolist()
	train_base_label = labels[:train_base_size]
	train_online_label = labels[train_base_size:train_base_online_size]
	testing_label = labels[train_base_online_size:]
	print('knowledge base size: {}, online learning: {}, testing size: {}'.format(train_base_data.shape,
			train_online_data.shape, testing_data.shape))

	fail_index_list = np.nonzero(train_base_label)[0]
	succ_index_list = [i for i in range(train_base_size) if i not in fail_index_list]
	return succ_index_list, fail_index_list, train_base_data, train_online_data, testing_data, train_base_label, train_online_label, testing_label


def clustering(para, index_list, selected_data):
	data_dist = pdist(selected_data, metric=compute_dist)
	Z = linkage(data_dist, 'complete')
	cluster_results = fcluster(Z, para['max_d'], criterion='distance')
	clus_num = len(set(cluster_results))
	print('There are %d clusters in this initial clustering' % (clus_num))
	index_per_clu = [[] for _ in range(clus_num)]  # initialization
	for i, clu in enumerate(cluster_results):
		index_per_clu[clu-1].append(index_list[i])
	return cluster_results, index_per_clu


def extract_repre(train_base_data, index_per_clu, dis_sum_list):
	represents = []
	for indexes_clu in index_per_clu:
		each_cluster_data = train_base_data[indexes_clu, :]
		dist_matrix = get_dis_mat(np.array(each_cluster_data))
		min_score = float('inf')
		min_index = -1
		for cl in range(len(each_cluster_data)):
			index = indexes_clu[cl]
			score = np.sum(dist_matrix[cl]) - dist_matrix[cl, cl]
			dis_sum_list[index] = score    # keep the distance sum between the index and all others
			if score < min_score:
				min_index = index
				min_score = score
		represents.append(train_base_data[min_index])
	return represents


def anomalyDetect(para, succ_index_list, fail_index_list, train_base_data, train_online_data, testing_data, train_online_label, testing_label, total_inst_num ):
	fail_data = train_base_data[fail_index_list,:]
	succ_dta = train_base_data[succ_index_list,:]
	print(fail_data.shape, succ_dta.shape)
	assert fail_data.shape[0] + succ_dta.shape[0] == train_base_data.shape[0]

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
	assert len(testing_label) == len(predict_label)
	ev.evaluate(testing_label, predict_label)


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


def get_dis_mat(data):
	inst_num, event_num = data.shape
	dist_mat =-1*np.ones((inst_num, inst_num))
	for i in range(inst_num):
		for j in range(i, inst_num):
			dist_mat[i, j] = dist_mat[j, i] = compute_dist(data[i, :], data[j, :])
	return dist_mat


def compute_dist(Sei,Sej):
	# in our case, the distance lies in 0 ~ 1, where 1 represents exactly the same.
	weigh=float(LA.norm(Sei)*LA.norm(Sej))
	if weigh ==0:
		return 1
	result = 1 - np.dot(Sei,Sej)/weigh
	if abs(result) < 1e-8:
		result = 0
	return result


def detect(para, fail_repre_list, succ_repre_list, testing_data):
	prediction=[-1] * testing_data.shape[0]
	for r, row in enumerate(testing_data):
		dist_list = []
		for fail_repre in fail_repre_list:
			dist_list.append(compute_dist(fail_repre,row))
		if min(dist_list) < para['fail_threshold']:
			prediction[r] = 1
		else:
			succ_dist_list=[]
			for succ_repre in succ_repre_list:
				succ_dist_list.append(compute_dist(succ_repre, row))
			if min(succ_dist_list) < para['succ_threshold']:
				prediction[r] = 0
			else:
				prediction[r] = 1
	return prediction
