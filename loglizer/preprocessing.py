"""
The interface for data preprocessing.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
import sys
from collections import Counter
from scipy.special import expit

class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None):
        """ Fit and transform the data matrix

        Arguments
        ---------
            X: log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`

        Returns
        -------
            X_new: The transformed data matrix
        """
        self.term_weighting = term_weighting
        self.normalization = normalization
        X_df = pd.DataFrame()
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            for event, count in event_counts.items():
                X_df.loc[i, event] = count
        self.events  = X_df.columns
        X = X_df.fillna(0).values

        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(float(num_instance) / df_vec)
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        # elif self.term_weighting == 'tf-idf-sigmoid':
        #     df_vec = np.sum(X > 0, axis=0)
        #     self.idf_vec = np.log(float(num_instance) / df_vec)
        #     idf_matrix = X * expit(np.tile(self.idf_vec, (num_instance, 1)))
        #     X = idf_matrix
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X
        print('====== Transformed train data summary ======')
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 
        return X_new

    def transform(self, X_seq):
        """ Transform the data matrix with trained parameters

        Arguments
        ---------
            X: log sequences matrix
            term_weighting: None or `tf-idf`

        Returns
        -------
            X_new: The transformed data matrix
        """
        X_df = pd.DataFrame(columns=self.events)
        for i in range(X_seq.shape[0]):
            X_df.loc[i, :] = [0] * len(self.events)
            event_counts = Counter(X_seq[i])
            for event, count in event_counts.items():
                if event in self.events:
                    X_df.loc[i, event] = count
        X = X_df.fillna(0).values
        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        # elif self.term_weighting == 'tf-idf-sigmoid':
        #     idf_matrix = X * expit(np.tile(self.idf_vec, (num_instance, 1)))
        #     X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X
        print('====== Transformed test data summary ======')
        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 

        return X_new
