"""
The interface to load Thunderbird log datasets.

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
    df['Time'] = pd.to_datetime(df['Time'], format="%Y-%m-%d-%H.%M.%S.%f")
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
