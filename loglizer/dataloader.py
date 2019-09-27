"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
import sys
from sklearn.utils import shuffle
from collections import OrderedDict

def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]

    # fixed shuffle ---- in order to have the same result
    indexes = shuffle(np.arange(x_train.shape[0]), random_state=7)
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)

def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential',
    save_csv=False):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print('====== Input data summary ======')

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, train_ratio, split_type)

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None)
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)


def load_BGL(log_file, label_file=None, window='sliding', time_interval=60, stepping_size=60, 
             train_ratio=0.8):
    """  TODO

    """



def bgl_preprocess_data(para, raw_data, event_mapping_data):
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
    sliding_file_path = para['save_path']+'_sliding_'+str(para['window_size'])+'h_'+str(para['step_size'])+'h.csv'

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    # get the list of label data and the list of time data
    label_data, time_data = raw_data[:,0], raw_data[:,1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        # get the first value in the time_data list
        start_time = time_data[0]
        print("the start_time is:",start_time)
        print("the type of time is:",type(start_time))
        # the index points at the index in the time_data list
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            # the start_time + para['window_size']:
            ## start_time is the first value in the time_data list
            ## get the data scope using the window size
            ## cur_time < the result means it is in the scope of window size
            print("the current time is:",cur_time)
            # if cur_time < start_time + para['window_size']*3600:
            if int(cur_time) < int(start_time) + para['window_size'] * 3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index,end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        while end_index < log_size:
            # start_time = start_time + para['step_size']*3600
            # end_time = end_time + para['step_size']*3600
            start_time = int(start_time) + para['step_size']*3600
            end_time = int(end_time) + para['step_size']*3600
            for i in range(start_index,end_index):
                # if time_data[i] < start_time:
                if int(time_data[i]) < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                # if time_data[j] < end_time:
                if int(time_data[j]) < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset\n'%inst_number)
        np.savetxt(sliding_file_path, start_end_index_list, delimiter=',', fmt='%d')
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
    print("the event_mapping_data is:", event_mapping_data)
    event_num = len(list(set(event_mapping_data)))
    print('There are %d log events'%event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    # inst_number --- row, every row is a log sequence(windows sliding)
    # event_num --- column, every column is a event, the number is the occurrence of a corresponding event
    event_count_matrix = np.zeros((inst_number,event_num))
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            print("the length of expanded_indexes_list is:",len(expanded_indexes_list[j]))
            print("the k value is:",k)
            event_index = event_mapping_data[k]
            print("the event_index is:", event_index)
            # the index is not different from the eventId
            event_index = event_index-1
            event_count_matrix[j, event_index] += 1
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies"%sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    return event_count_matrix, labels



# this is a part of test for bgl_preprocess_data function
# import os
# import pandas as pd
# import numpy as np
# from collections import Counter
#
# para = {}
# para['save_path'] = '../../logparser-master/logs/BGL/BGL_2k.log_matrix'
# para['window_size'] = 24 # 24 hours ---- one day
# para['step_size'] = 3 # 3 hours
#
# # list data, the element is tuple of (label, time)
#
# # System log Detection/Anomaly_Detection_Time.ipynb
# df_raw_data = pd.read_csv('../../logparser-master/logs/BGL/BGL_2k.log_structured.csv')
# raw_data = []
# for label, time in zip(df_raw_data['Label'],df_raw_data['Timestamp']):
#     raw_data.append((label, time))
# # raw_data
# raw_data = np.array(raw_data)
#
# df_map_event = pd.read_csv('../../logparser-master/logs/BGL/BGL_2k.log_structured.csv')
# event_mapping_data = []
# ids = []
# ids = [int(x[1:]) for x in df_map_event['EventId']]
#
# for id, log in zip(ids, df_map_event['EventTemplate']):
#     event_mapping_data.append([id,log])
#
#
# event_count_matrix, labels = bgl_preprocess_data(para, raw_data, event_mapping_data)
# print("the event_count_matrix is:", Counter(event_count_matrix[9]))
# print("the labels are:", Counter(labels))


def load_Linux(log_file, label_file=None, window ='sliding', time_interval = None,stepping_size = None, train_ratio = 0.5, split_type = 'sequential', save_csv=False):

    print('========== Input data summary==========')
    if log_file.endswith('.npy'):
        # split training and validation set in a class-uniform way
        assert window == 'sliding','Only window=session is supported for Linux dataset'

        data_df = np.load(log_file)
        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential','Warning: Only split type=sequential is supported'
            # split training and validation set sequentially
            x_data = data_df
            (x_train,_),(x_test,_) = _split_data(x_data, train_ratio = train_ratio, split_type = split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(x_data.shape[0], x_train.shape[0], x_test.shape[0]))

            return (x_train, None), (x_test, None)
    else:
        raise NotImplementedError('load_Linux() only support npy files')

# this is a part of test for linux_preprocess_data function --- get the event matrix


def Linux_preprocess_data(para, raw_data, event_mapping_data):
    """
    split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (Time) --- we will transfer the time to seconds, and get the abs
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = para['save_path']+'_sliding_'+str(para['window_size'])+'h_'+str(para['step_size'])+'h.csv'
    print("the sliding_file_path is:", sliding_file_path)

    # ============= divide into sliding windows ============

    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    # get the list of label data and the list of time data
    time_data = raw_data

    if not os.path.exists(sliding_file_path):
        start_time = time_data[0]
        start_index = 0
        end_index = 0
        # finish the comparision in one roll with window_size
        for cur_time in time_data:
            if cur_time < start_time + para['window_size'] * 3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair = tuple((start_index, end_index))
                start_end_index_list.append(start_end_pair)
                break
        # sliding the block and change the index of start and end
        while end_index < log_size:
            # add the sliding size to start time
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index, end_index):
                if time_data[i] < start_time:
                    i += 1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j += 1
                else:
                    break
            start_index = i
            end_index = j
            # update the start_end_pair
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        # compute how many sequence(lines) in total
        inst_number = len(start_end_index_list)
        print("there are %d instances (sliding windows) in this dataset"%(inst_number))
        np.savetxt(sliding_file_path, start_end_index_list, delimiter=',', fmt='%d')
    else:
        print("Loading start_end_index_list from file")
        start_end_index_list = pd.read_csv(sliding_file_path, header = None).values
        inst_number = len(start_end_index_list)
        print("there are %d instances (sliding windows) in this dataset"%(inst_number))

    # get all the log indexes in each time window by ranging from start_index to end_index
    # in order to counter
    expanded_indexes_list = []
    for t in range(inst_number):
        # for every row(sequence), there should be a index_list
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        # get the index_list for every row
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        # add the indexes for a sequence to expanded_indexed_list
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = [row[0] for row in event_mapping_data]
    # get the total number for events
    event_num = len(list(set(event_mapping_data)))
    print("the event number is:", event_num)

    # ============ get event count of each sliding window =============
    event_count_matrix = np.zeros((inst_number, event_num))
    for j in range(inst_number):
        for k in expanded_indexes_list[j]:
            event_index = event_mapping_data[k]
            # make the eventId suitable for list index
            event_index = event_index - 1
            event_count_matrix[j, event_index] += 1

    return event_count_matrix





