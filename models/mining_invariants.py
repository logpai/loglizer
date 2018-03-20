#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

import numpy as np
from itertools import combinations
import utils.evaluation as ev

def estimate_invar_spce(para, event_count_matrix):
	""" Estimate the Invariant Space using SVD decomposition, return the invariant space size r (integer)

	Args:
	--------
	para: the parameter dictionary
	event_count_matrix: the event count matrix (each row is a log sequence vector, each column represents an event)

	Returns:
	--------
	invar_size: invariant space size
	"""

	dot_result = np.dot(event_count_matrix.T, event_count_matrix)/float(event_count_matrix.shape[0])
	u, s, v = np.linalg.svd(dot_result)  # SVD decomposition
	print("SVD decomposition results (U,S,V) size: ", u.shape, s.shape, v.shape)
	# Since the sigular values are in descending order, we start from the right most column of matrix v
	inst_num, event_num = event_count_matrix.shape
	for i in range(event_num-1, -1, -1):
		col_vec = v[i, :]
		count = sum(abs(np.dot(event_count_matrix, col_vec)) < para['epsilon'])
		if count < inst_num * para['threshold']:
			print('from the %d(not included) column to the rightmost column, they are validated invariant' % i)
			break
	invar_size = event_num - i - 1
	print('The size of invariants should be %d' % invar_size)
	return invar_size


def compute_eigenvector(event_count_matrix):
	""" calculate the smallest eigenvalue and corresponding eigenvector (theta in the paper) for a given sub_matrix

	Args:
	--------
	event_count_matrix: the event count matrix (each row is a log sequence vector, each column represents an event)

	Returns:
	--------
	min_vec: the eigenvector of corresponding minimum eigen value
	FLAG_contain_zero: whether the min_vec contains zero (very small value)
	"""

	FLAG_contain_zero = False
	count_zero = 0
	dot_result = np.dot(event_count_matrix.T, event_count_matrix)
	u, s, v = np.linalg.svd(dot_result)
	min_vec = v[-1, :]
	for i in min_vec:
		if np.fabs(i) < 1e-6:
			count_zero += 1
	if count_zero != 0:
		print("0 exits and discard the following vector: ")
		FLAG_contain_zero=True
	min_vec = min_vec.T
	return min_vec, FLAG_contain_zero


def check_invar_validity(para, event_count_matrix, selected_columns):
	""" scale the eigenvector of float number into integer, and check whether the scaled number is valid

	Args:
	--------
	para: the parameter dictionary
	event_count_matrix: the event count matrix (each row is a log sequence vector, each column represents an event)
	selected_columns: select columns from all column list

	Returns:
	--------
	validity: whether the selected columns is valid
	scaled_theta: the scaled theta vector
	"""

	sub_matrix = event_count_matrix[:, selected_columns]
	inst_num = event_count_matrix.shape[0]
	validity = False
	print ('selected matrix columns are', selected_columns)
	min_theta, FLAG_contain_zero = compute_eigenvector(sub_matrix)
	abs_min_theta = [np.fabs(it) for it in min_theta]
	if FLAG_contain_zero:
		return validity, []
	else:
		for i in para['scale_list']:
			min_index = np.argmin(abs_min_theta)
			scale = float(i) / min_theta[min_index]
			scaled_theta = np.array([round(item * scale) for item in min_theta])
			scaled_theta[min_index] = i
			scaled_theta = scaled_theta.T
			if 0 in np.fabs(scaled_theta):
				continue
			dot_submat_theta = np.dot(sub_matrix, scaled_theta)
			count_zero = 0
			for j in dot_submat_theta:
				if np.fabs(j) < 1e-8:
					count_zero += 1
			if count_zero >= para['threshold'] * inst_num:
				validity = True
				print('A valid invariant is found and the corresponding columns are ',scaled_theta, selected_columns)
				break
		return validity, scaled_theta


def invariant_search(para, event_count_matrix, invar_size):
	""" Generate invariant candidates, check the validity, return the obtained invariants

	Args:
	--------
	para: the parameter dictionary
	event_count_matrix: the event count matrix (each row is a log sequence vector, each column represents an event)
	invar_size: invariant space size

	Returns:
	--------
	invar_dict: dictionary of invariants, where selected columns is the key, and invariant is the value
	"""

	num_samples, num_features = event_count_matrix.shape
	invar_dict = dict()	 # save the mined Invariants(value) and its corresponding columns(key)
	search_space = []  # only invariant candidates in this list are valid.

	# invariant of only one column (all zero columns)
	init_cols = sorted([[item] for item in range(num_features)])
	for col in init_cols:
		search_space.append(col)
	init_col_list = init_cols[:]
	for col in init_cols:
		if np.count_nonzero(event_count_matrix[:, col]) == 0:
			invar_dict[frozenset(col)] = [1]
			search_space.remove(col)
			init_col_list.remove(col)
	print('the remaining features are: ', init_col_list)

	item_list = init_col_list
	length = 2
	FLAG_break_loop = False
	# check invariant of more columns
	while len(item_list) != 0:
		if para['stop_invar_num'] != 'None':
			if len(item_list[0]) >= para['stop_invar_num']:
				break
		joined_item_list = join_set(item_list, length)    # generate new invariant candidates
		for items in joined_item_list:
			if check_candi_valid(items, length, search_space):
				search_space.append(items)
		item_list = []
		for item in joined_item_list:
			if frozenset(item) in invar_dict.keys():
				continue
			if item not in search_space:
				continue
			if not check_candi_valid(frozenset(item), length, search_space) and length > 2:
				search_space.remove(item)
				continue 	# an item must be superset of all other subitems in searchSpace, else skip
			validity, scaled_theta = check_invar_validity(para, event_count_matrix, item)
			if validity:
				prune(invar_dict.keys(), set(item), search_space)
				invar_dict[frozenset(item)] = scaled_theta
				search_space.remove(item)
			else:
				item_list.append(item)
			if len(invar_dict) >= invar_size:
				FLAG_break_loop = True
				break
		if FLAG_break_loop:
			break
		length += 1
	return invar_dict


def prune(valid_cols, new_item_set, search_space):
	""" prune invalid combination of columns

	Args:
	--------
	valid_cols: existing valid column list
	new_item_set: item set to be merged
	search_space: the search space that stores possible candidates

	Returns:
	--------
	"""

	if len(valid_cols) == 0:
		return
	for se in valid_cols:
		intersection = set(se) & new_item_set
		if len(intersection) == 0:
			continue
		union = set(se) | new_item_set
		for item in list(intersection):
			diff = sorted(list(union - set([item])))
			if diff in search_space:
				search_space.remove(diff)


def join_set(item_list, length):
	""" Join a set with itself and returns the n-element (length) itemsets

	Args:
	--------
	item_list: current list of columns
	length: generate new items of length

	Returns:
	--------
	return_list: list of items of length-element
	"""

	set_len = len(item_list)
	return_list = []
	for i in range(set_len):
		for j in range(i+1, set_len):
			i_set = set(item_list[i])
			j_set = set(item_list[j])
			if len(i_set.union(j_set)) == length:
				joined = sorted(list(i_set.union(j_set)))
				if joined not in return_list:
					return_list.append(joined)
	return_list = sorted(return_list)
	return return_list


def check_candi_valid(item, length, search_space):
	""" check whether an item's subitems are in searchspace

	Args:
	--------
	item: item to be checked
	length: the length of item
	search_space: the search space that stores possible candidates

	Returns:
	--------
	Valid or not: True or False
	"""

	for subItem in combinations(item, length - 1):
		if sorted(list(subItem)) not in search_space:
			return False
	return True


def evaluate(event_count_matrix, invar_dict, groundtruth_labels):
	""" evaluate the results with mined invariants

	Args:
	--------
	event_count_matrix: the input event count matrix
	invar_dict: the dictionary of invariants
	groundtruth_labels: the groundtruth labels for evaluation

	Returns:
	--------
	"""
	print("the mined {} invariants are: {}".format(len(invar_dict), invar_dict))
	valid_col_list = []
	valid_invar_list = []
	for key in invar_dict:
		valid_col_list.append(list(key))
		valid_invar_list.append(list(invar_dict[key]))

	prediction = []
	for row in event_count_matrix:
		label = 0
		for i, cols in enumerate(valid_col_list):
			sum_of_invar = 0
			for j, c in enumerate(cols):
				sum_of_invar += valid_invar_list[i][j] * row[c]
			if sum_of_invar != 0:
				label = 1
				break
		prediction.append(label)

	assert len(groundtruth_labels) == len(prediction)
	ev.evaluate(groundtruth_labels, prediction)
