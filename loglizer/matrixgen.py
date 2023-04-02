import os
import pandas as pd
import numpy as np
from collections import Counter
import re
from dataloader import *
import joblib
import optparse

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
    para['save_path'] = '../../Dataset_ML/Linux/Client/Client_train/'
    para['window_size'] = 0.5 # 24 hours ---- one day
    para['step_size'] = 0.2 # 3 hours

    # =============================== generate the event matrix for norcom linux logs =========================

    # set the format of command input
    parser = optparse.OptionParser('usage %prog --p1 <structured log filename for training data> \
                                    --p2 <dict_filename> --p3 <structured id log filename for training data> --p4 <transformed matrix for training> \
                                    --p5 <structured log filename for testing data> --p6 <dict_filename_com> --p7 <structured id log filename for testing data> \
                                    --p8 <transformed matrix for testing>')
    # set the elements for every parameter
    parser.add_option('--p1', dest='structured_log_filename', type='string', help='Please input the structured log filename: ')
    parser.add_option('--p2', dest='dict_filename', type='string', help='Please input the dict filename for training data: ')
    parser.add_option('--p3', dest='structured_log_id_filename', type='string', help='Please input the structured log id filename: ')
    parser.add_option('--p4', dest='matrix', type='string', help='Please input the location where you want to save the matrix: ')
    parser.add_option('--p5', dest='structured_log_com_filename', type='string', help='Please input the coming structured log filename: ')
    parser.add_option('--p6', dest='dict_filename_com', type='string', help='Please input the dict filename for testing data')
    parser.add_option('--p7', dest='structured_log_id_com_filename', type='string', help='Please input the coming structured log id filename: ')
    parser.add_option('--p8', dest='matrix_com', type='string', help='Please input the location where you want to save the coming matrix: ')


    # parser arguments through the parse_args()
    (options, args) = parser.parse_args()
    # get the values from options
    structured_log_filename = options.structured_log_filename
    dict_filename = options.dict_filename
    structured_log_id_filename = options.structured_log_id_filename
    matrix = options.matrix
    structured_log_com_filename = options.structured_log_com_filename
    dict_filename_com = options.dict_filename_com
    structured_log_id_com_filename = options.structured_log_id_com_filename
    matrix_com = options.matrix_com

    # get the linux dataframe
    fd_linux = pd.read_csv(structured_log_filename)
    # make a copy to avoid modifying the original data
    fd_linux = fd_linux.copy()

    # dict_filename has been given by parser
    # check whether the dict_filename has existed
    if os.path.isfile(dict_filename):
        event_map = joblib.load(dict_filename)
    else:
        event_map = Event_Convert(fd_linux, dict_filename)
    # shift the key and value of the dict
    event_map = {val: key for (key, val) in event_map.items()}
    
    #for i in range(len(fd_linux['EventId'])):
     #   for key, value in event_map.items():
      #      # print("the key {} and value {}".format(key,  value))
        #    if fd_linux['EventId'][i] == value:
       #         # replace the hashed eventId into format like numerical id
         #       fd_linux.is_copy = False
          #      fd_linux['EventId'][i] = key
           #     print("the replace eventId is:", fd_linux['EventId'][i])
    

    #fd_linux['EventId'].map(event_map).fillna(fd_linux['EventId'])
    fd_linux['EventId'] = fd_linux['EventId'].map(event_map)

    # structured_log_id_filename has been generated above
    
    
    fd_linux.to_csv(structured_log_id_filename, index = False)
    # read the saved csv
    fd_linux_id = pd.read_csv(structured_log_id_filename)
    # sort the dataframe from time increasing order
    fd_linux_id_sort = fd_linux_id.copy()
    fd_linux_id_sort.sort_index(axis=0, ascending=False, inplace=True)
    # reset the index
    fd_linux_id_sort = fd_linux_id_sort.reset_index(drop = True)
    print(fd_linux_id_sort.head())
    # part to transform the month, date, time into seconds
    month_list, time_list, day_list, day_list = [], [], [], []

    for i in range(len(fd_linux_id_sort['Time'])):
        time_list.append(fd_linux_id_sort['Time'][i].split(':'))
    for j in range(len(fd_linux_id_sort['Date'])):
        day_list.append(fd_linux_id_sort['Date'][j])

    month_number = 0
    for k in range(len(fd_linux_id_sort['Month'])):
        month_number = month_string_to_number(fd_linux_id_sort['Month'][k])
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
    # matrix path has been generated above
    np.save(matrix, event_count_matrix)



    # =============================== generate the event matrix for comicious linux logs =========================

    para_com = {}
    para_com['save_path'] = '../../Dataset_ML/Linux/Client/Client_com/'
    para_com['window_size'] = 24  # 24 hours ---- one day
    para_com['step_size'] = 3  # 3 hours

    # structured_log_com_filename has been give by parser
    fd_linux_com = pd.read_csv(structured_log_com_filename)
    fd_linux_com = fd_linux_com.copy()

    # dict_filename_com has been given by parser
    # check whether the dict_filename_com has existed
    if os.path.isfile(dict_filename_com):
        event_map_com = joblib.load(dict_filename_com)
    else:
        event_map_com = Event_Convert(fd_linux_com, dict_filename_com)

    for i in range(len(fd_linux_com['EventId'])):
        for key, value in event_map_com.items():
            fd_linux_com.is_copy = False
            if fd_linux_com['EventId'][i] == value:
                fd_linux_com['EventId'][i] = key

    # structured_log_com_filename
    fd_linux_com.to_csv(structured_log_id_com_filename, index=False)

    fd_linux_com_id = pd.read_csv(structured_log_id_com_filename)
    fd_linux_com_id = fd_linux_com_id.copy()

    fd_linux_com_id.sort_index(axis=0, ascending=False, inplace=True)

    fd_linux_com_id = fd_linux_com_id.reset_index(drop = True)

    fd_linux_com_id = fd_linux_com_id.copy()

    # part to transform date time into seconds
    month_list_com ,time_list_com, day_list_com, day_list_com = [],[],[], []

    for i in range(len(fd_linux_com_id['Time'])):
        time_list_com.append(fd_linux_com_id['Time'][i].split(':'))
    for j in range(len(fd_linux_com_id['Date'])):
        day_list_com.append(fd_linux_com_id['Date'][j])

    month_number_com = 0
    for k in range(len(fd_linux_com_id['Month'])):
        # print("we are transferring the month:",fd_linux['Month'][k])
        month_number_com = month_string_to_number(fd_linux_com_id['Month'][k])
        month_list_com.append(month_number_com)

    seconds_list_com = trans_seconds(month_list_com, day_list_com, time_list_com)

    raw_data_com = np.array(seconds_list_com)

    event_mapping_data_com = []
    Event_ids_com = []
    # get the digits part of eventID
    Event_ids_com = [int(x) for x in fd_linux_com_id['EventId']]

    for id, log in zip(Event_ids_com, fd_linux_com_id['EventTemplate']):
        event_mapping_data_com.append([id, log])


    event_count_matrix_com = Linux_preprocess_data(para_com, raw_data_com, event_mapping_data_com)
    # print("the event_count_matrix is:", Counter(event_count_matrix[9]))
    print("the event_count_matrix is:", event_count_matrix_com)
    # matrix_com has been given by parser
    np.save(matrix_com, event_count_matrix_com)
