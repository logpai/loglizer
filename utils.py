import pandas as pd
import os
import pickle
import gc
from tqdm import tqdm

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


def load_data(structure_file, pkl_path):
    parsed_result = pd.read_csv(structure_file, nrows=1)
    content2tempalte = dict(zip(parsed_result["Content"], parsed_result["EventTemplate"]))

    with open(os.path.join(pkl_path, "train_small.pkl"), "rb") as fr:
        train_windows = pickle.load(fr)[0:1]
    with open(os.path.join(pkl_path, "test_small.pkl"), "rb") as fr:
        test_windows = pickle.load(fr)[0:1]

    # with open(os.path.join(pkl_path, "train_small.pkl"), "wb") as fw:
    #     pickle.dump(train_windows, fw)
    # with open(os.path.join(pkl_path, "test_small.pkl"), "wb") as fw:
    #     pickle.dump(test_windows, fw)

    train_x, train_y = get_x_y(train_windows, content2tempalte)
    test_x, test_y = get_x_y(test_windows, content2tempalte)

    del parsed_result, content2tempalte
    gc.collect()

    return train_x, train_y, test_x, test_y