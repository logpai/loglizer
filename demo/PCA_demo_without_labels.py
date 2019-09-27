#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' This is a demo file for the PCA model.
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
from loglizer.models import PCA
from loglizer import dataloader, preprocessing
from collections import Counter
import pandas as pd

# struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
struct_log = '../../Dataset_ML/Linux_matrix/log_matrix.npy'
mal_struct_log = '../../Dataset_ML/Linux_mal_matrix/mal_matrix.npy'


if __name__ == '__main__':
    # # 1. Load structured log file and extract feature vectors
    # # Save the raw event sequence file by setting save_csv=True
    # (x_train, _), (_, _) = dataloader.load_HDFS(struct_log, window='session',
    #                                             split_type='sequential', save_csv=True)
    # feature_extractor = preprocessing.FeatureExtractor()
    # x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',
    #                                           normalization='zero-mean')
    #
    # ## 2. Train an unsupervised model
    # print('Train phase:')
    # # Initialize PCA, or other unsupervised models, LogClustering, InvariantsMiner
    # model = PCA()
    # # Model hyper-parameters may be sensitive to log data, here we use the default for demo
    # model.fit(x_train)
    # # Make predictions and manually check for correctness. Details may need to go into the raw logs
    # y_train = model.predict(x_train)
    #
    # ## 3. Use the trained model for online anomaly detection
    # print('Test phase:')
    # # Load another new log file. Here we use struct_log for demo only
    # (x_test, _), (_, _) = dataloader.load_HDFS(struct_log, window='session', split_type='sequential')
    # # Go through the same feature extraction process with training, using transform() instead
    # x_test = feature_extractor.transform(x_test)
    # # Finally make predictions and alter on anomaly cases
    # y_test = model.predict(x_test)
    # print("the result is:",y_test)
    # print("the labels are:",Counter(y_test))


    # example without train_ratio
    (x_train, _), (_, _) = dataloader.load_Linux(struct_log, window='sliding',split_type='sequential', save_csv = True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', normalization='zero-mean')

    # 2.Train an unsupervised model
    print("Train phase")
    # Initialize PCA
    model = PCA()
    # model hyper-parameters may be sensitive to log data, here we use the default for demo
    model.fit(x_train)
    # make predictions and manually check for correctness. Details may need to go into the raw logs
    y_train = model.predict(x_train)

    # 3. Use the trained model for online anomaly detection
    print("Test phase:")
    # load another new log file, here we should know the basic set should be large as much as possible
    # cuz for every vector, the same position may have different meanings --- can not be compared
    (x_test,_),(_,_) = dataloader.load_Linux(mal_struct_log, window = 'sliding', split_type = 'sequential')
    # go through the same feature extraction process with training

    x_test_original = x_test.copy()
    # assert x_test == x_train, 'the training data is not the same with testing data'
    x_test = feature_extractor.transform(x_test)
    # Finally make predictions and alter on anomaly cases
    y_test = model.predict(x_test)
    # build the tracing dict
    x_y_dict = {}
    # define the counter
    i = 0
    for x,y in zip(x_test_original, y_test):
        x_y_dict[str(x)+','+str(i)] = y_test
        i += 1
    # print("the result is:", len(y_test))
    # print("the key names are:", x_y_dict.keys())
    # get the indexs of anomaly sequences
    anomaly_sequence_index = [i for i in range(len(y_test)) if y_test[i] == 1]
    print("the index of anomaly sequence is:", anomaly_sequence_index)

    # trace the index in the sliding_file_path
    sliding_file_path = '../../Dataset_ML/Linux_mal_sliding_24h_3h.csv'
    for index in anomaly_sequence_index:
        # read sliding file: start_end_index
        fd = pd.read_csv(sliding_file_path, header = None)
        start_index, end_index = None, None
        # get the start and end time from index value
        start_index = fd.iloc[index,:][0]
        end_index = fd.iloc[index,:][1]
        print("please check log csv indexes between {} and {}".format(start_index, end_index))

    anomaly_sequence = []
    for index in anomaly_sequence_index:
        # anomaly_sequence = [var for var in x_y_dict.keys() if int(var.split(',')[-1]) == index]

        for var in x_y_dict.keys():
            # print("the var is:",var)
            if int(var.split(',')[-1]) == index:
                # print out the anomaly test_x sequence
                # print(var)
                anomaly_sequence.append(var)

    # print("the anomaly sequence is:", len(anomaly_sequence))
    print("the lables are:", Counter(y_test))
    print("the counter is {} and the anomaly rate is: {}".format(Counter(y_test), len(anomaly_sequence)/x_test.shape[0]))

'''
For HDFS:
the result is: [0. 0. 0. ... 0. 0. 0.]
the labels are: Counter({0.0: 3951, 1.0: 19}) --- there are 19 anomalies
For Linux_logs:
Counter({0.0: 163, 1.0: 3/5})   0.0184 --- 0.0307
For Linux_mali_logs:
Counter({0.0: 127, 1.0: 25})    0.1969
'''




