#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import cross_validate


def cv_evaluate(clf, train_data, train_labels):
	""" evaluation with cross validation in classifiers, by default: 10-CV.
		report the precision, recall, and f1-measure
	Args:
	--------
	clf: the classifier
	train_data: the training data
	train_labels: the training labels

	Returns:
	--------
	"""
	print("Using cross validation")
	scoring = ['precision', 'recall', 'f1']
	scores_cv = cross_validate(clf, train_data, train_labels.ravel(), cv=10, scoring=scoring)
	precision, recall, f1_score = np.mean([scores_cv['test_precision'], scores_cv['test_recall'], scores_cv['test_f1']], axis=1)
	print('=' * 20, 'RESULT', '=' * 20)
	print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))


def evaluate(testing_labels, prediction):
	""" evaluation with precision, recall, and f1-measure.

	Args:
	--------
	testing_labels: labels of testing data
	prediction: predicted labels of a model

	Returns:
	--------
	"""
	precision, recall, f1_score, _ = np.array(list(precision_recall_fscore_support(testing_labels, prediction)))[:, 1]
	print('=' * 20, 'RESULT', '=' * 20)
	print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))