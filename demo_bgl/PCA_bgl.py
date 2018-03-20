#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

from models import PCA as PCA
from utils import data_loader as data_loader

para={
'path':'../../Data/BGL_data/', # directory for input data
'log_file_name':'BGL_MERGED.log', # filename for log data file
'log_event_mapping':'logTemplateMap.csv', # filename for log-event mapping. A list of event index, where each row represents a log
'save_path': '../time_windows/', # dir for saving sliding window data files to avoid splitting
'select_column':[0,4], # select the corresponding columns (label and time) in the raw log file
'window_size':3,  # time window (unit: hour)
'step_size': 1,  # step size (unit: hour)
'tf-idf': False, # tf-idf should set to false in BGL data since it can get better accuracy
'fraction':0.95
}

if __name__ == '__main__':
	raw_data, event_mapping_data = data_loader.bgl_data_loader(para)
	event_count_matrix, labels = data_loader.bgl_preprocess_data(para, raw_data, event_mapping_data)
	weigh_data = PCA.weighting(para, event_count_matrix)
	threshold, C = PCA.get_threshold(para, weigh_data)
	PCA.anomaly_detection(weigh_data, labels, C, threshold)


