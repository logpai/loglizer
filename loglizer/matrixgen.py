import os
import pandas as pd
import numpy as np
from collections import Counter
import re
from dataloader import *
import joblib

# function to transform hours and minutes to seconds
def trans_seconds(time_list):
    seconds_list = []
    seconds = 0
    for i in range(len(time_list)):
        #         print("splitting time:",time_list[i])
        seconds = int(time_list[i][0]) * 3600 + int(time_list[i][1]) * 60 + int(time_list[i][2])
        seconds_list.append(seconds)
    return seconds_list

# transformation between month name to numbers
def month_string_to_number(string):
    m = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }
    s = string.strip()[:3]

    try:
        out = m[s]
        return out
    except:
        pattern = '<.*>(.*)'
        match = re.match(pattern,string)
        s = match.group(1)
        out = m[s]
        return out
        # process the special case with <N/ASCII>Jun
        # raise ValueError('Not a month')

# transform month, day to seconds
def trans_seconds(month_list, day_list, time_list):
    seconds_list = []
    seconds = 0
    for i in range(len(day_list)):
        # we assume there are 30 days for every month
        seconds = (int(month_list[i]) - int(month_list[0])) * 30 * 24 * 3600 + (int(day_list[i]) - int(day_list[0])) * 24 * 3600 + \
                  int(time_list[i][0]) * 3600 + int(time_list[i][1]) * 60 + int(time_list[i][2])
        # print("the seconds are:", seconds)
        seconds_list.append(seconds)
    return seconds_list

# transform log key to eventID
# def Event_Convert(fd):
#     event_map = {}
#     for i, event in enumerate(fd['EventId']):
#         event_map['E' + str(i+1)] = event
#
#     return event_map
def Event_Convert(fd, filename):
    event_map = {}
    event_list = None
    event_list = fd['EventId']
    # get the unique values in a list
    event_list = list(set(event_list))
    for i, event in enumerate(event_list):
        event_map[str(i+1)] = event
    joblib.dump(event_map, filename)
    return event_map


if __name__ == "__main__":

    # define the window_size and step_size to get time sequence
    para = {}
    para['save_path'] = '../../Dataset_ML/Linux'
    para['window_size'] = 24 # 24 hours ---- one day
    para['step_size'] = 3 # 3 hours

    # =============================== generate the event matrix for normal linux logs =========================
    # get the linux dataframe
    fd_linux = pd.read_csv('../../Dataset_ML/Linux_2k.log_structured.csv')
    # make a copy to avoid modifying the original data
    fd_linux = fd_linux.copy()

    filename = '../../Dataset_ML/Linux_matrix/Event_dict.pkl'
    # check whether the event_dict has existed
    if os.path.isfile(filename):
        event_map = joblib.load(filename)
    else:
        event_map = Event_Convert(fd_linux, filename)

    for i in range(len(fd_linux['EventId'])):
        for key, value in event_map.items():
            fd_linux.is_copy = False
            if fd_linux['EventId'][i] == value:
                fd_linux['EventId'][i] = key

    fd_linux.to_csv('../../Dataset_ML/Linux_2k.log_structured_id.csv', index=0)

    fd_linux_id = pd.read_csv('../../Dataset_ML/Linux_2k.log_structured_id.csv')
    fd_linux_id = fd_linux_id.copy()

    # part to transform the month, date, time into seconds
    month_list, time_list, day_list, day_list = [], [], [],[]

    for i in range(len(fd_linux_id['Time'])):
        time_list.append(fd_linux_id['Time'][i].split(':'))
    for j in range(len(fd_linux_id['Date'])):
        day_list.append(fd_linux_id['Date'][j])

    month_number = 0
    for k in range(len(fd_linux_id['Month'])):
        # print("we are transferring the month:",fd_linux['Month'][k])
        month_number = month_string_to_number(fd_linux_id['Month'][k])
        month_list.append(month_number)

    seconds_list = trans_seconds(month_list, day_list, time_list)

    raw_data = np.array(seconds_list)

    event_mapping_data = []
    Event_ids = []
    # get the digits part of eventID
    Event_ids = [int(x) for x in fd_linux_id['EventId']]

    for id, log in zip(Event_ids, fd_linux_id['EventTemplate']):
        event_mapping_data.append([id, log])


    # create the event count matrix with the function of Linux_preprocess_data
    event_count_matrix = Linux_preprocess_data(para, raw_data, event_mapping_data)
    # print("the event_count_matrix is:", Counter(event_count_matrix[9]))
    print("the event_count_matrix is:", event_count_matrix)
    matrix = '../../Dataset_ML/Linux_matrix/log_matrix.npy'
    np.save(matrix, event_count_matrix)
    # np.load(matrix+'.npy')


    # =============================== generate the event matrix for malicious linux logs =========================

    para_mal = {}
    para_mal['save_path'] = '../../Dataset_ML/Linux_mal'
    para_mal['window_size'] = 24  # 24 hours ---- one day
    para_mal['step_size'] = 3  # 3 hours

    fd_linux_mali = pd.read_csv('../../Dataset_ML/malicious_linux.log_structured.csv')
    fd_linux_mali = fd_linux_mali.copy()

    filename_mali = '../../Dataset_ML/Linux_mal_matrix/Event_mal_dict.pkl'
    # check whether the event_dict has existed
    if os.path.isfile(filename_mali):
        event_map_mal = joblib.load(filename_mali)
    else:
        event_map_mal = Event_Convert(fd_linux_mali, filename_mali)

    for i in range(len(fd_linux_mali['EventId'])):
        for key, value in event_map_mal.items():
            fd_linux_mali.is_copy = False
            if fd_linux_mali['EventId'][i] == value:
                fd_linux_mali['EventId'][i] = key

    fd_linux_mali.to_csv('../../Dataset_ML/malicious_linux.log_structured_id.csv', index=0)

    fd_linux_mali_id = pd.read_csv('../../Dataset_ML/malicious_linux.log_structured_id.csv')
    fd_linux_mali_id = fd_linux_mali_id.copy()

    # part to transform date time into seconds
    month_list_mal ,time_list_mal, day_list_mal, day_list_mal = [],[],[], []

    for i in range(len(fd_linux_mali_id['Time'])):
        time_list_mal.append(fd_linux_mali_id['Time'][i].split(':'))
    for j in range(len(fd_linux_mali_id['Date'])):
        day_list_mal.append(fd_linux_mali_id['Date'][j])

    month_number_mal = 0
    for k in range(len(fd_linux_mali_id['Month'])):
        # print("we are transferring the month:",fd_linux['Month'][k])
        month_number_mal = month_string_to_number(fd_linux_mali_id['Month'][k])
        month_list_mal.append(month_number_mal)

    seconds_list_mal = trans_seconds(month_list_mal, day_list_mal, time_list_mal)

    raw_data_mal = np.array(seconds_list_mal)

    event_mapping_data_mal = []
    Event_ids_mal = []
    # get the digits part of eventID
    Event_ids_mal = [int(x) for x in fd_linux_mali_id['EventId']]

    for id, log in zip(Event_ids_mal, fd_linux_mali_id['EventTemplate']):
        event_mapping_data_mal.append([id, log])


    event_count_matrix_mal = Linux_preprocess_data(para_mal, raw_data_mal, event_mapping_data_mal)
    # print("the event_count_matrix is:", Counter(event_count_matrix[9]))
    print("the event_count_matrix is:", event_count_matrix_mal)
    mal_matrix = '../../Dataset_ML/Linux_mal_matrix/mal_matrix.npy'
    np.save(mal_matrix, event_count_matrix_mal)
    # np.load(mal_matrix)