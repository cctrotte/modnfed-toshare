import torch
import wandb
import os
from modules_modn import ModularModel
from modules_baseline import BaselineModel
from utils.utils_epoct import get_data_for_nodes
 
from dotenv import load_dotenv
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.plot_and_save import load_config
from utils.utils_epoct import get_numeric_data
from utils.plots import plot_heatmap
import random
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier



from sklearn.metrics import (
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)



load_dotenv()

config = load_config(config_file="config_epoct_cv.yaml")

with open("node_splits.pkl", "rb") as f:
    node_splits = pickle.load(f)


seed = config["seed"]
random.seed(42)


feature_types = load_config(config_file="epoct_feature_type.yaml")
target_types = load_config(config_file="epoct_target_type.yaml")

#model_dir = "trained_models/modn.pt"
model_dir = "cv_models/final_30small/saved_models_bsl/state_dim=30__hidden_layers=16,16,16__mu=0.0__0.pt"
checkpoint = torch.load(model_dir)

df = pd.read_csv("./datasets/epoct_plus_cleaned.csv")
df_num = get_numeric_data(df, feature_types, target_types)

model_type = "bsl"
all = True

# Reconstruct and load models from the checkpoint
types = ["fl", "local", "centralized"]
models = {type:{} for type in types}
for type in types:
    for node_id in checkpoint[type].keys():
        model_config = checkpoint[type][node_id]['config']
        state_dict = checkpoint[type][node_id]['state_dict']
        if model_type == 'modn':
            model = ModularModel(**model_config)
        else:
            model = BaselineModel(**model_config)
        model.load_state_dict(state_dict)
        models[type][node_id] = model

ids_ = sorted([models['fl'][i+1].node_id for i in range(len(models["fl"]))])

print(f'{sum([checkpoint["fl"][node_id]["auprc"] > checkpoint["local"][node_id]["auprc"] for node_id in checkpoint["fl"].keys()])/len(checkpoint["fl"])}')

node_variables = {"node_" + str(i): {"health_facility_id": models["fl"][ix+1].node_id, "features": sorted(models["fl"][ix+1].feature_names), "targets": sorted(list(models['fl'][1].target_names_and_types.keys()))} for ix, i in enumerate(sorted(ids_))}

health_facilities = [elem['health_facility_id'] for k, elem in node_variables.items()]

all_features = sorted(
    list(set([f for elem in node_variables.values() for f in elem["features"]]))
)
all_targets = sorted(
    list(set([f for elem in node_variables.values() for f in elem["targets"]]))
)

# use last fold for validation
kf = KFold(n_splits=5, shuffle=False,)
splits_per_node = {
n: list(kf.split(np.array(node_splits[n]["train"])))
    for n in ids_}

node_splits_fold = {}
fold = 0
for n in sorted(ids_):

    train_ids = np.array(node_splits[n]["train"])
    test_ids  = node_splits[n]["test"]

    idx_tr, idx_val = splits_per_node[n][fold]

    # build per-fold splits for all nodes
    node_splits_fold[n] = {"train": train_ids[idx_tr].tolist(), "valid": train_ids[idx_val].tolist(), "test": test_ids}

(
    X_centralized_train,
    X_centralized_valid,
    y_centralized_train,
    y_centralized_valid,
    node_data_train,
    node_data_valid,
    node_data_test,
    feature_index_mapping,
    target_index_mapping,
    n_samples,
) = get_data_for_nodes(
    all_features,
    all_targets,
    feature_types,
    target_types,
    node_variables,
    df_num,
    seed,
    model_type,
    split_ids = node_splits_fold,
)
print("--")



# select a node

aus = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}
f1s = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}

fl_better = []

print("Training local RandomForest models on node-specific data...")
trained_locals = {}
for i, node_idx in enumerate(ids_):
    print(node_idx)
    # Instantiate sklearn RandomForestClassifier
    rf = rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    # Prepare node training data
    X_train = node_data_train[i][0]
    y_train = node_data_train[i][1]
    # Fit RandomForest
    rf.fit(X_train, y_train)
    trained_locals[node_idx] = rf

print("Local RandomForest training complete.")

for node_idx in range(1, len(node_data_valid)+1):

    facility_id = health_facilities[node_idx-1]
    X_full = df_num[df_num["health_facility_id"] == facility_id]
    model_fl = models["fl"][node_idx]
    #model_local = models["local"][node_idx]
    model_local = trained_locals[facility_id]
    model_cl = models["centralized"][node_idx]
    X_valid = node_data_valid[node_idx-1][0]
    y_valid = node_data_valid[node_idx-1][1]
    X_valid_mask = ~node_data_valid[node_idx-1][4]
    y_valid_mask = ~node_data_valid[node_idx-1][5]
    model_fl.eval()
    #model_local.eval()
    model_cl.eval()

    # compute some performance metrics
    preds_fl = torch.sigmoid(model_fl(X_valid)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_fl(X_valid)[1]).detach() 
    preds_fl_class = (preds_fl >= 0.5).float()

    # preds_local =  torch.sigmoid(model_local(X_valid)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_local(X_valid)[1]).detach() 
    # preds_local_class = (preds_local >= 0.5).float()

    proba_list = model_local.predict_proba(X_valid)
    # For multi-output, predict_proba returns list of arrays (n_samples, n_classes)

    fixed_probas = []
    for c, p in zip(model_local.classes_, proba_list):
                     # numpy array, e.g. [0] or [1] or [0,1]
        if p.shape[1] == 1:
            if c[0] == 0:             # now classes[0] is a scalar
                p = np.vstack([p[:,0], 1-p[:,0]]).T
            else:
                p = np.vstack([1-p[:,0], p[:,0]]).T
        fixed_probas.append(p)


    # now fixed_probas[i] is always (n_samples, 2) and you can do:
    preds_local = np.vstack([p[:,1] for p in fixed_probas]).T
    preds_local_class = (preds_local >= 0.5).astype(float)

    preds_cl = torch.sigmoid(model_cl(X_valid)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_cl(X_valid)[1]).detach()
    preds_cl_class = (preds_cl >= 0.5).float()

    fl_better_than_local = 0
    num_tars = 0

    for i, t in enumerate(sorted(list(model_fl.target_names_and_types.keys()))):
        target = y_valid[:, model_fl.output_dims[t]]

        if len(np.unique(target)) > 1:
            num_tars += 1
            aus["fl"][t].append(average_precision_score(target, preds_fl[:,i]))
            f1s["fl"][t].append(f1_score(target, preds_fl_class[:,i], average="macro"))

            aus["local"][t].append(average_precision_score(target, preds_local[:,i]))
            f1s["local"][t].append(f1_score(target, preds_local_class[:,i], average="macro"))


            if average_precision_score(target, preds_fl[:,i]) >= average_precision_score(target, preds_local[:,i]):
                fl_better_than_local += 1
    
        
    fl_better.append(fl_better_than_local/num_tars)

    for i,t in enumerate(sorted(list(model_cl.target_names_and_types.keys()))):
        if t in list(model_fl.target_names_and_types.keys()):
            target = y_valid[:, model_cl.output_dims[t]]

            if len(np.unique(target)) > 1:
                aus["centralized"][t].append(average_precision_score(target, preds_cl[:,i]))
                f1s["centralized"][t].append(f1_score(target, preds_cl_class[:,i], average="macro"))





print(sum([True if elem > 0.5 else False for elem in fl_better ])/len(fl_better))

aus_mean = {type: {t: np.mean(aus[type][t]) for t in aus[type]} for type in types}
f1s_mean = {type: {t: np.mean(f1s[type][t]) for t in f1s[type]} for type in types}
#
print(f'FL: aus mean {np.mean(list(aus_mean["fl"].values()))} f1 mean {np.mean(list(f1s_mean["fl"].values()))}')
print(f'Local: aus mean {np.mean(list(aus_mean["local"].values()))} f1 mean {np.mean(list(f1s_mean["local"].values()))}')
print(f'Centralized: aus mean {np.mean(list(aus_mean["centralized"].values()))} f1 mean {np.mean(list(f1s_mean["centralized"].values()))}')




print("End of script")
