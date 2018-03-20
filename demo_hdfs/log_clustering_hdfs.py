#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Shilin He'

from utils import data_loader as data_loader
from models import log_clustering as cluster

para={
'path':'../../Data/data_differentSize/SOSP_diffSize/',
'log_seq_file_name':'SOSP_12m_Vectors.txt', # filename for log sequence data file
'label_file_name':'SOSP_12m_labels.txt', # filename for label data file
'train_base_per': 0.6,
'train_online_per': 0.2,
'max_d': 0.3,                   # the threshold for cutoff the cluster process
'repre_threshold':0.2,
'fail_threshold': 0.1,   
'succ_threshold':0.99,

}

if __name__ == '__main__':
	raw_data, label_data = data_loader.hdfs_data_loader(para)
	weighted_matrix, total_inst_num = cluster.weighting(raw_data)
	succ_index_list, fail_index_list, train_base_data, train_online_data, testing_data, train_base_label, train_online_label, testing_label = cluster.split_data(para,weighted_matrix,label_data)
	cluster.anomalyDetect(para, succ_index_list, fail_index_list, train_base_data, train_online_data, testing_data, train_online_label, testing_label, total_inst_num)
