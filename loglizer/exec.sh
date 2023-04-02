#!/bin/bash

path1='../../Dataset_ML/Linux/Client/Client_train/structured_log.csv'
path2='../../Dataset_ML/Linux/Client/Client_train/Event_dict.pkl'
path3='../../Dataset_ML/Linux/Client/Client_train/structured_log_id.csv'
path4='../../Dataset_ML/Linux/Client/Client_train/Linux_matrix/log_matrix.npy'
path5='../../Dataset_ML/Linux/Client/Client_com/structured_log.csv'
path6='../../Dataset_ML/Linux/Client/Client_com/Event_dict.pkl'
path7='../../Dataset_ML/Linux/Client/Client_com/structured_log_id.csv'
path8='../../Dataset_ML/Linux/Client/Client_com/Linux_matrix/log_matrix.npy'

python3 matrixgen_client.py --p1 $path1 --p2 $path2 --p3 $path3 --p4 $path4 --p5 $path5 --p6 $path6 --p7 $path7 --p8 $path8

exit 0
