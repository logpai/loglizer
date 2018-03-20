#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

from utils import data_loader as data_loader
from models import classifiers as cl

para={
'path':'../../Data/SOSP_data/', # directory for input data
'log_seq_file_name':'rm_repeat_rawTFVector.txt', # filename for log sequence data file
'label_file_name':'rm_repeat_mlabel.txt', # filename for label data file
'training_percent': 0.8, # training data percentage
'models': 'SVM', # select from ['DT', 'LR', 'SVM']
'cross_validate': False # set to True to avoid over_fitting (10-CV), but if we want to predict anomalies, it should set to False, Default: False
}


if __name__ == '__main__':
	model = para['models']
	assert model in ['DT', 'LR', 'SVM']
	raw_data, label_data = data_loader.hdfs_data_loader(para)
	train_data, train_labels, testing_data, testing_labels = cl.data_split(para, raw_data, label_data)
	# Select one models out of three provided models
	if model == 'DT':
		cl.decision_tree(para, train_data, train_labels, testing_data, testing_labels)
	elif model == 'LR':
		cl.logsitic_regression(para, train_data, train_labels, testing_data, testing_labels)
	elif model == 'SVM':
		cl.SVM(para, train_data, train_labels, testing_data, testing_labels)