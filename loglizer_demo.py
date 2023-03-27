#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
from loglizer.models import PCA, IsolationForest, DecisionTree, LR
from loglizer import dataloader, preprocessing
import pickle
from tqdm import tqdm
from sklearn.ensemble import IsolationForest as iForest
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import time
import tracemalloc


def get_x_y(windows, content2tempalte):
    x = []
    y = []
    for window in tqdm(windows):
        template_list = []
        y_list = []
        for item in window:
            template = content2tempalte.get(item["Content"], "")
            template_list.append(template)
            y_list.append(item["Label"])
        x.append(template_list)
        y.append(1 if sum(y_list) > 0 else 0)
    return x, y 


model_name = "if"
dataname = "BGL"

if __name__ == "__main__":
    config_info = {
        "BGL": {
            "structure_file": "../../data/BGL/BGL.log_structured.csv",
            "pkl_path": "../../proceeded_data/BGL/BGL_ws=60m_s=60m_0.8train",
        },
        "Thunderbird": {
            "structure_file": "../../src/Drain_result/Thunderbird.log_structured.csv",
            "pkl_path": "../../proceeded_data/Thunderbird/Thunderbird_ws=60m_s=60m_0.8train",
        },
    }

    structure_file = config_info[dataname]["structure_file"]
    pkl_path = config_info[dataname]["pkl_path"]

    parsed_result = pd.read_csv(structure_file)
    content2tempalte = dict(zip(parsed_result["Content"], parsed_result["EventTemplate"]))

    with open(os.path.join(pkl_path, "train.pkl"), "rb") as fr:
        train_windows = pickle.load(fr)[0:1]
    with open(os.path.join(pkl_path, "test.pkl"), "rb") as fr:
        test_windows = pickle.load(fr)[0:1]
    train_x, train_y = get_x_y(train_windows, content2tempalte)
    test_x, test_y = get_x_y(test_windows, content2tempalte)

    del parsed_result, content2tempalte

    feature_extractor = preprocessing.FeatureExtractor()
    

    if model_name.lower() == "if":
        model = iForest(n_estimators=100, max_features=1)

        s1 = time.time()
        train_feat = feature_extractor.fit_transform(np.array(train_x), term_weighting='tf-idf', 
                                                normalization='zero-mean')
        model.fit(train_feat)
        s2 = time.time()
        
        pred_train = model.decision_function(train_feat)
        proba_train = (pred_train-pred_train.min()) / (pred_train.max() - pred_train.min())

        s3 = time.time()
        test_feat = feature_extractor.transform(np.array(test_x))
        pred_test = model.decision_function(test_feat)
        s4 = time.time()
        proba_test = (pred_test-pred_test.min()) / (pred_test.max() - pred_test.min())

    elif model_name.lower() == "dt":
        
        s1 = time.time()
        train_feat = feature_extractor.fit_transform(np.array(train_x), term_weighting='tf-idf', 
                                                normalization='zero-mean')
        model = DecisionTree()
        model.fit(train_feat, train_y)
        s2 = time.time()

        proba_train = model.predict_proba(train_feat)[:, 1]
        
        s3 = time.time()
        test_feat = feature_extractor.transform(np.array(test_x))
        proba_test = model.predict_proba(test_feat)[:, 1]
        s4 = time.time()

    elif model_name.lower() == "lr":
        s1 = time.time()
        train_feat = feature_extractor.fit_transform(np.array(train_x), term_weighting='tf-idf', 
                                                normalization='zero-mean')
        model = LR()
        model.fit(train_feat, train_y)
        s2 = time.time()

        proba_train = model.predict_proba(train_feat)[:, 1]
        
        s3 = time.time()
        test_feat = feature_extractor.transform(np.array(test_x))
        proba_test = model.predict_proba(test_feat)[:, 1]
        s4 = time.time()

    print("Training time for {}: {:.3f}".format(model_name, s2-s1))
    print("Testing time for {}: {:.3f}".format(model_name, s4-s3))
    # print(f"Peak memory usage: {tracemalloc.get_traced_memory()[1] / (1024*1024):.2f} MB")
