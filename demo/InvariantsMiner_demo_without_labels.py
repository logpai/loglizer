#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' This is a demo file for the Invariants Mining model.
    API usage:
        dataloader.load_HDFS(): load HDFS dataset
        feature_extractor.fit_transform(): fit and transform features
        feature_extractor.transform(): feature transform after fitting
        model.fit(): fit the model
        model.predict(): predict anomalies on given data
        model.evaluate(): evaluate model accuracy with labeled data
'''

import sys
sys.path.append('../')
from loglizer.models import InvariantsMiner
from loglizer import dataloader, preprocessing

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file
epsilon = 0.5 # threshold for estimating invariant space

if __name__ == '__main__':
    # Load structured log without label info
    (x_train, _), (x_test, _) = dataloader.load_HDFS(struct_log,
                                                     window='session', 
                                                     train_ratio=0.5,
                                                     split_type='sequential')
    # Feature extraction
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)

    # Model initialization and training
    model = InvariantsMiner(epsilon=epsilon)
    model.fit(x_train)

    # Predict anomalies on the training set offline, and manually check for correctness
    y_train = model.predict(x_train)

    # Predict anomalies on the test set to simulate the online mode
    # x_test may be loaded from another log file
    x_test = feature_extractor.transform(x_test)
    y_test = model.predict(x_test)

    # If you have labeled data, you can evaluate the accuracy of the model as well.
    # Load structured log with label info
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                               label_file=label_file,
                                                               window='session', 
                                                               train_ratio=0.5,
                                                               split_type='sequential')   
    x_test = feature_extractor.transform(x_test)
    precision, recall, f1 = model.evaluate(x_test, y_test)

