#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

import numpy as np
import math
import utils.evaluation as ev


def weighting(para, raw_data):
	""" weight the matrix using idf

	Args:
	--------
	raw_data: log sequences matrix

	Returns:
	--------
	weigh_data: weighted raw data
	"""
	inst_size, event_num = raw_data.shape
	if para['tf-idf']:
		idf_data = np.zeros((inst_size, event_num))
		for j in range(event_num):
			cnt = np.count_nonzero(raw_data[:, j])
			if cnt != 0:
				for i in range(inst_size):
					idf_data[i, j] = raw_data[i, j]*math.log(inst_size/float(cnt))
		raw_data = idf_data
	raw_data = raw_data.T
	mean_value = raw_data.mean(axis=1)
	weigh_data = np.zeros((event_num, inst_size))
	for i in range(inst_size):
		weigh_data[:, i] = raw_data[:, i] - mean_value
	return weigh_data


def get_threshold(para, weigh_data):
	""" compute the threshold for anomaly detection

	Args:
	--------
	weigh_data: weighted raw data

	Returns:
	--------
	threshold: used as the threshold, which determines the anomalies
	C: the projection matrix
	"""
	U, sigma, VT = np.linalg.svd(weigh_data, full_matrices=False)
	event_num, inst_size = weigh_data.shape

	# calculate the number of principal components
	k=4
	tot = np.dot(sigma,sigma)
	tmp=0
	for i in range(len(sigma)):
		tmp += sigma[i] * sigma[i]
		if (tmp/tot) >= para['fraction']:
			break
	k = i+1
	print('principal components number is %d' % (k))

	for i in range(event_num):
		sigma[i] = sigma[i] * sigma[i]/float(inst_size)

	phi = np.zeros(3)
	for i in range(3):
		for j in range(k,event_num):
			phi[i] += math.pow(sigma[j],i+1)

	h0 = 1-2*phi[0]*phi[2]/(3*phi[1]*phi[1])
	c_alpha = 3.2905
	threshold = phi[0]*math.pow((c_alpha*math.sqrt(2*phi[1]*h0*h0)/phi[0] + 1.0 + phi[1]*h0*(h0-1)/phi[0]/phi[0]), (1.0/h0))
	P = U[:,:k]
	I = np.identity(event_num,int)
	C = I - np.dot(P,P.transpose())
	return threshold, C

def anomaly_detection(weigh_data, label_data, C, threshold):
	""" detect anomalies by projecting into a subspace with C

	Args:
	--------
	weigh_data: weighted raw data
	label_data: the labels list
	threshold: used as the threshold, which determines the anomalies
	C: the projection matrix

	Returns:
	--------

	"""
	print ('there are %d anomalies' % (sum(label_data)))
	event_num, inst_size  = weigh_data.shape
	predict_results = np.zeros((inst_size),int)
	print('the threshold is %f' % (threshold))
	for i in range(inst_size):
		ya = np.dot(C,weigh_data[:,i])
		SPE = np.dot(ya,ya)
		if SPE > threshold:
			predict_results[i] = 1	#1 represent failure
	assert len(label_data) == len(predict_results)
	ev.evaluate(label_data, predict_results)
