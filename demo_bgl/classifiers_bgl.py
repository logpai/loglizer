#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

from utils import data_loader as data_loader
from models import classifiers as cl

para={
'path':'../../Data/BGL_data/', # directory for input data
'log_file_name':'BGL_MERGED.log', # filename for log data file
'log_event_mapping':'logTemplateMap.csv', # filename for log-event mapping. A list of event index, where each row represents a log
'save_path': '../time_windows/', # dir for saving sliding window data files to avoid splitting
'select_column':[0,4], # select the corresponding columns (label and time) in the raw log file
'window_size':3,  # time window (unit: hour)
'step_size': 1,  # step size (unit: hour)
'training_percent': 0.8,  # training data percentage
'tf-idf': True, # whether to use tf-idf
'models': 'DT', # select from ['DT', 'LR', 'SVM']
'cross_validate': False # set to True to avoid over_fitting (10-CV), but if we want to predict anomalies, it should set to False, Default: False
}


if __name__ == '__main__':
	model = para['models']
	assert model in ['DT', 'LR', 'SVM']
	raw_data, event_mapping_data = data_loader.bgl_data_loader(para)
	event_count_matrix, labels = data_loader.bgl_preprocess_data(para, raw_data, event_mapping_data)
	train_data, train_labels, testing_data, testing_labels = cl.data_split(para, event_count_matrix, labels)
	# Select one models out of three provided models
	if model == 'DT':
		cl.decision_tree(para, train_data, train_labels, testing_data, testing_labels)
	elif model == 'LR':
		cl.logsitic_regression(para, train_data, train_labels, testing_data, testing_labels)
	elif model == 'SVM':
		cl.SVM(para, train_data, train_labels, testing_data, testing_labels)


