#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import PCA, InvariantsMiner
from loglizer import dataloader, preprocessing

run_models = ['PCA', 'InvariantsMiner']
struct_log = '../data/HDFS/HDFS.npz' # The benchmark dataset

if __name__ == '__main__':
    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,
                                                           train_ratio=0.5)
    
    for _model in run_models:
        print('Evaluating {} on HDFS:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', 
                                                      normalization='zero-mean')
            model = PCA()
            model.fit(x_train)
        
        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train[y_train == 0, :]) # Use only normal samples for training
        
        x_test = feature_extractor.transform(x_te)
        print('Train:')
        precision, recall, f1 = model.evaluate(x_train, y_train)
        print('Test:')
        precision, recall, f1 = model.evaluate(x_test, y_test)


