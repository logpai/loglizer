#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

from utils import data_loader as data_loader
from models import mining_invariants as mi

para = {
'path':'../../Data/SOSP_data/',        # directory for input data
'log_seq_file_name':'rm_repeat_rawTFVector.txt', # filename for log sequence data file
'label_file_name':'rm_repeat_mlabel.txt', # filename for label data file
'epsilon':2.0,                          # threshold for the step of estimating invariant space
'threshold':0.98,                       # percentage of vector Xj in matrix satisfies the condition that |Xj*Vi|<epsilon
'scale_list':[1,2,3],					# list used to sacle the theta of float into integer
'stop_invar_num':3                      # stop if the invariant length is larger than stop_invar_num. None if not given
}

if __name__ == '__main__':
	raw_data, label_data = data_loader.hdfs_data_loader(para)
	r = mi.estimate_invar_spce(para, raw_data)
	invar_dict = mi.invariant_search(para, raw_data, r)
	mi.evaluate(raw_data, invar_dict, label_data)
