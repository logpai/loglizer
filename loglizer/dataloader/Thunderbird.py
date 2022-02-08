"""
The interface to load HDFS log datasets.

Authors:
    Hans Aschenloher
"""

import random
import pandas as pd
import os
import numpy as np
import re
from sklearn.utils import shuffle
from collections import OrderedDict

def loadDataset(log_file, window='sliding', time_interval=60, stepping_size=30,
             train_ratio=0.8):
    """ Read a Thunderbird log file to obtain training and test data.

    Args:
    --------
    log_file: Input file name with the header
        "LineId,Label,Timestamp,,,,,,,,,EventId,,", where commas stand for
        unnecessary fields.
    windows: Type of windows to use. Can be either 'sliding' or 'fixed'.
    time_interval: Time scope of a window in seconds. Used for both fixed and
        sliding windows.
    stepping_size: Step size of sliding windows in seconds. Used only for
        sliding windows.
    train_ratio: Fraction of examples to use for training.

    Returns
    -------
        (x_train, y_train): The training data.
        (x_test, y_test): The testing data.

    """

    # Load the file and sort lines according to time.
    df = pd.read_csv(log_file)
    #df['Time'] = pd.to_datetime( str(df['Month'])+" " + str(df['Day']) + " " + str(df['Time']), format="%b %d %H:%M:%S")
    df = df.sort_values(by="Timestamp")
    df.reset_index(drop=True, inplace=True)
    df['LineId'] = range(0, df.shape[0])

    examples = [] # List of sequences and anomaly labels.

    start_time = df['Timestamp'][0]
    end_time = df['Timestamp'].iloc[-1]

    assert window == 'fixed' or window == 'sliding', "Unsupported window."
    index = 0
    t0 = start_time
    t1 = t0 + time_interval
    while t1 < end_time:
        sequence = []
        is_anomaly = 0
        # Make a sequence and label it as normal or abnormal.
        while df['Timestamp'][index] < t1:
            sequence.append(df['EventId'][index])
            if df['Label'][index] != '-':
                is_anomaly = 1
            index += 1
        if sequence:
            examples.append([sequence, is_anomaly])
        # Translate the window.
        if window == "fixed":
            t0 = t1
        elif window == "sliding":
            t0 += stepping_size
        t1 = t0 + time_interval

    random.shuffle(examples)
    x = [t[0] for t in examples]
    y = [t[1] for t in examples]

    n_train = int(len(x) * train_ratio)

    x_train = np.array(x[:n_train], dtype=list)
    y_train = np.array(y[:n_train], dtype=int)
    x_test  = np.array(x[n_train:], dtype=list)
    y_test  = np.array(y[n_train:], dtype=int)

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(len(y), sum(y), len(y) - sum(y)))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(len(y_train), sum(y_train), len(y_train) - sum(y_train)))
    print('Test: {} instances, {} anomaly, {} normal' \
          .format(len(y_test), sum(y_test), len(y_test) - sum(y_test)))

    return (x_train, y_train), (x_test, y_test)

def thunderbird_preprocess_data(para, raw_data, event_mapping_data):
    """ split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = para['save_path']+'sliding_'+str(para['window_size'])+'h_'+str(para['step_size'])+'h.csv'

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    label_data, time_data = raw_data[:,0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if  cur_time < start_time + para['window_size']*3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index,end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        while end_index < log_size:
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index,end_index):
                if time_data[i] < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset\n'%inst_number)
        np.savetxt(sliding_file_path,start_end_index_list,delimiter=',',fmt='%d')
    else:
        print('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list=[]
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = [row[0] for row in event_mapping_data]
    event_num = len(list(set(event_mapping_data)))
    print('There are %d log events'%event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    event_count_matrix = np.zeros((inst_number,event_num))
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            event_index = event_mapping_data[k]
            event_count_matrix[j, event_index] += 1
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies"%sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    return event_count_matrix, labels
