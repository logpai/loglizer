#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Shilin He'

from models import PCA as PCA
from utils import data_loader as data_loader

para = {
'path':'../../Data/SOSP_data/', # directory for input data
'log_seq_file_name':'rm_repeat_rawTFVector.txt', # filename for log sequence data file
'label_file_name':'rm_repeat_mlabel.txt', # filename for label data file
'fraction':0.95
}


if __name__ == '__main__':
	raw_data, label_data = data_loader.hdfs_data_loader(para)
	weigh_data = PCA.weighting(raw_data)
	threshold, C = PCA.get_threshold(para, weigh_data)
	PCA.anomaly_detection(weigh_data, label_data, C, threshold)


