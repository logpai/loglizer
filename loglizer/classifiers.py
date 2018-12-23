#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import math
import utils.evaluation as ev

def data_split(para, raw_data, label_data):
	""" split data into training set and testing set according to training percentage defined in para.

	Args:
	--------
	para: the parameters dictionary
	raw_data: log sequence matrix
	label_data: labels

	Returns:
	--------
	train_data: log sequence matrix for training
	train_labels: labels for training
	testing_data: log sequence matrix for testing/evaluation
	testing_labels: labels for testing/evaluation
	"""

	inst_num, event_num = raw_data.shape
	traing_size = int(math.floor(inst_num * para['training_percent']))
	print('Training size is %d while testing size is %d' % (traing_size, inst_num-traing_size))
	train_data = raw_data[:traing_size]
	train_labels = label_data[:traing_size]
	testing_data = raw_data[traing_size:]
	testing_labels = label_data[traing_size:]
	return train_data, train_labels, testing_data, testing_labels


def decision_tree(para, train_data, train_labels, testing_data, testing_labels):
	""" train a decision tree models and evaluate on testing data

	Args:
	--------
	train_data: log sequence matrix for training
	train_labels: labels for training
	testing_data: log sequence matrix for testing/evaluation
	testing_labels: labels for testing/evaluation

	Returns:
	--------
	precision: The evaluation metric Precision
	recall: The evaluation metric Recall
	f1_score: The evaluation metric F1_score
	"""

	print("Train a Decision Tree Model")
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(train_data, train_labels)
	prediction = list(clf.predict(testing_data))
	assert len(prediction) == len(testing_labels)

	if para['cross_validate']:
		ev.cv_evaluate(clf, train_data, train_labels)
	else:
		ev.evaluate(testing_labels, prediction)

def logsitic_regression(para, train_data, train_labels, testing_data, testing_labels):
	""" train a logistic regression models and evaluate on testing data

	Args:
	--------
	train_data: log sequence matrix for training
	train_labels: labels for training
	testing_data: log sequence matrix for testing/evaluation
	testing_labels: labels for testing/evaluation

	Returns:
	--------
	precision: The evaluation metric Precision
	recall: The evaluation metric Recall
	f1_score: The evaluation metric F1_score
	"""

	print("Train a Logistic Regression Model")
	clf = LogisticRegression(C=100, penalty='l1', tol=0.01, class_weight='balanced', multi_class='ovr')
	clf = clf.fit(train_data, train_labels.ravel())
	prediction = list(clf.predict(testing_data))
	assert len(prediction) == len(testing_labels)

	if para['cross_validate']:
		ev.cv_evaluate(clf, train_data, train_labels)
	else:
		ev.evaluate(testing_labels, prediction)


def SVM(para, train_data, train_labels, testing_data, testing_labels):
	""" train a support vector machine models and evaluate on testing data

	Args:
	--------
	train_data: log sequence matrix for training
	train_labels: labels for training
	testing_data: log sequence matrix for testing/evaluation
	testing_labels: labels for testing/evaluation

	Returns:
	--------
	precision: The evaluation metric Precision
	recall: The evaluation metric Recall
	f1_score: The evaluation metric F1_score
	"""

	print("Train a SVM Model")
	clf = svm.LinearSVC(penalty='l1', tol=0.0001, C=1, dual=False, fit_intercept=True, intercept_scaling=1, class_weight='balanced', max_iter=1000)
	clf = clf.fit(train_data, train_labels.ravel())
	prediction = list(clf.predict(testing_data))
	assert len(prediction) == len(testing_labels)

	if para['cross_validate']:
		ev.cv_evaluate(clf, train_data, train_labels)
	else:
		ev.evaluate(testing_labels, prediction)
