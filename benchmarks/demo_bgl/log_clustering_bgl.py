#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Shilin He'

from utils import data_loader as data_loader
from models import log_clustering as cluster

para={
'path':'../../Data/BGL_data/', # directory for input data
'log_file_name':'BGL_MERGED.log', # filename for log data file
'log_event_mapping':'logTemplateMap.csv', # filename for log-event mapping. A list of event index, where each row represents a log
'save_path': '../time_windows/', # dir for saving sliding window data files to avoid splitting
'select_column':[0,4], # select the corresponding columns (label and time) in the raw log file
'window_size':3,  # time window (unit: hour)
'step_size': 1,  # step size (unit: hour)
'train_base_per': 0.6,
'train_online_per': 0.2,
'max_d': 0.8,                   # the threshold for cutoff the cluster process
'repre_threshold': 0.2,
'fail_threshold': 0.1,   
'succ_threshold': 0.99,
}


if __name__ == '__main__':
	raw_data, event_mapping_data = data_loader.bgl_data_loader(para)
	event_count_matrix, labels = data_loader.bgl_preprocess_data(para, raw_data, event_mapping_data)
	weighted_matrix, total_inst_num = cluster.weighting(event_count_matrix)
	succ_index_list, fail_index_list, train_base_data, train_online_data, testing_data, train_base_label, train_online_label, testing_label = cluster.split_data(para,weighted_matrix,labels)
	cluster.anomalyDetect(para, succ_index_list, fail_index_list, train_base_data, train_online_data, testing_data, train_online_label, testing_label, total_inst_num)
