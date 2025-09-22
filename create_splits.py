import pandas as pd
import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils.plot_and_save import *
from utils.utils_epoct import get_numeric_data
import pickle


if __name__ == "__main__":

    df = pd.read_csv("./datasets/epoct_plus_cleaned.csv")
    feature_types = load_config(config_file="epoct_feature_type.yaml")
    target_types = load_config(config_file="epoct_target_type.yaml")
    df_num = get_numeric_data(df, feature_types, target_types)
    ids_ = df_num["health_facility_id"].unique()
    seed = 42

    node_splits = {id_: {"train": [], "test": []} for id_ in ids_}
    np.random.seed(seed)

    for index, id_ in enumerate(ids_):
        X_full = df_num[df_num["health_facility_id"] == id_]
        unique_patients = X_full["patient_id"].unique()
        np.random.shuffle(unique_patients)

        train_ratio = 0.7

        n_patients = len(unique_patients)
        n_train = int(n_patients * train_ratio)
        n_test = n_patients - n_train

        train_patients, test_patients = (
            unique_patients[:n_train],
            unique_patients[n_train:],
        )
        node_splits[id_]["train"] = train_patients
        node_splits[id_]["test"] = test_patients

    # Save the splits as a pickle file
    with open("node_splits.pkl", "wb") as f:
        pickle.dump(node_splits, f)

