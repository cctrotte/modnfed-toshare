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

from sklearn.metrics import (
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", SettingWithCopyWarning)

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module="sklearn.utils.extmath"
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
model_dir = "cv_models/final30small_5f/saved_modn/state_dim=30__hidden_layers=16,16,16__mu=0.0__0.pt"

#model_dir = 'trained_models/trained_modn_30small_noshuff/best_models.pt'
checkpoint = torch.load(model_dir)

df = pd.read_csv("./datasets/epoct_plus_cleaned.csv")
df_num = get_numeric_data(df, feature_types, target_types)

model_type = "modn"
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

ids_ = sorted([models['fl'][k].node_id for k in models['fl'].keys() if k!= 'global'])
count_ = df_num["health_facility_id"].value_counts()


print(f'{sum([checkpoint["fl"][node_id]["auprc"] > checkpoint["local"][node_id]["auprc"] for node_id in range(1, len(ids_)+1)])/len(ids_)}')

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



aus = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}
f1s = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}

aus_ensemble = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}
f1s_ensemble = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}
fl_better = []
for node_idx in range(1, len(node_data_valid)+1):

    facility_id = health_facilities[node_idx-1]
    X_full = df_num[df_num["health_facility_id"] == facility_id]
    model_fl = models["fl"][node_idx]
    model_local = models["local"][node_idx]
    model_cl = models["centralized"][node_idx]
    X_valid = node_data_valid[node_idx-1][0]
    y_valid = node_data_valid[node_idx-1][1]
    X_valid_mask = ~node_data_valid[node_idx-1][4]
    y_valid_mask = ~node_data_valid[node_idx-1][5]

    features = node_data_valid[node_idx-1][2]
    model_fl.eval()
    model_local.eval()
    model_cl.eval()

    # compute some performance metrics
    preds_fl = torch.sigmoid(model_fl(X_valid)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_fl(X_valid)[1]).detach() 
    preds_fl_class = (preds_fl >= 0.5).float()

    preds_local =  torch.sigmoid(model_local(X_valid)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_local(X_valid)[1]).detach() 
    preds_local_class = (preds_local >= 0.5).float()

    preds_cl = torch.sigmoid(model_cl(X_valid)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_cl(X_valid)[1]).detach()
    preds_cl_class = (preds_cl >= 0.5).float()

    fl_better_than_local = 0
    num_tars = 0


    targets = {}
    for i, t in enumerate(sorted(list(model_fl.target_names_and_types.keys()))):
        target = y_valid[:, model_fl.output_dims[t]]

        if len(np.unique(target)) > 1:
            num_tars += 1
            targets[t] = {"auprc": {}, "f1": {}}
            aus["fl"][t].append(average_precision_score(target, preds_fl[:,i]))
            f1s["fl"][t].append(f1_score(target, preds_fl_class[:,i], average="macro"))

            aus["local"][t].append(average_precision_score(target, preds_local[:,i]))
            f1s["local"][t].append(f1_score(target, preds_local_class[:,i], average="macro"))

            targets[t]["auprc"]["fl"] = average_precision_score(target, preds_fl[:,i])
            targets[t]["auprc"]["local"] = average_precision_score(target, preds_local[:,i])

            targets[t]["f1"]["fl"] = f1_score(target, preds_fl_class[:,i], average="macro")
            targets[t]["f1"]["local"] = f1_score(target, preds_local_class[:,i], average="macro")


            if average_precision_score(target, preds_fl[:,i]) >= average_precision_score(target, preds_local[:,i]):
                fl_better_than_local += 1
    
        
    fl_better.append(fl_better_than_local/num_tars)

    for i,t in enumerate(sorted(list(model_cl.target_names_and_types.keys()))):
        if t in list(model_fl.target_names_and_types.keys()):
            target = y_valid[:, model_cl.output_dims[t]]

            if len(np.unique(target)) > 1:
                aus["centralized"][t].append(average_precision_score(target, preds_cl[:,i]))
                f1s["centralized"][t].append(f1_score(target, preds_cl_class[:,i], average="macro"))

                targets[t]["auprc"]["cl"] = average_precision_score(target, preds_cl[:,i])
                targets[t]["f1"]["cl"] = f1_score(target, preds_cl_class[:,i], average = "macro")
    




print(sum([True if elem > 0.5 else False for elem in fl_better ])/len(fl_better))

aus_mean = {type: {t: np.mean(aus[type][t]) for t in aus[type]} for type in types}
f1s_mean = {type: {t: np.mean(f1s[type][t]) for t in f1s[type]} for type in types}

#
print(f'FL: aus mean {np.mean(list(aus_mean["fl"].values()))} f1 mean {np.mean(list(f1s_mean["fl"].values()))}')
print(f'Local: aus mean {np.mean(list(aus_mean["local"].values()))} f1 mean {np.mean(list(f1s_mean["local"].values()))}')
print(f'Centralized: aus mean {np.mean(list(aus_mean["centralized"].values()))} f1 mean {np.mean(list(f1s_mean["centralized"].values()))}')




# feature importance
def apply_model_feat_subset(feats_to_keep, model, X_valid, i_patient=None, feat_order=None):
    if i_patient is None:
        tmp = torch.full(X_valid.shape, False, dtype = torch.bool)
    else:
        tmp = torch.full(X_valid[i_patient].shape, False, dtype=torch.bool)
    idx_to_keep = [model.input_dims[f] for f in feats_to_keep]
    idx_to_keep = [i for s in idx_to_keep for i in s]

    tmp[:,idx_to_keep] = True
    if i_patient is None:
        input_ = X_valid.clone()
    else:
        input_ = X_valid[i_patient,:].clone()
    input_[~tmp] = np.nan
    if i_patient is not None:
        input_ = input_.reshape(1,-1)
    _, pred, order = model(input_, feat_order = feat_order)
    pred = torch.sigmoid(pred).detach().numpy()
    if i_patient is not None:
        pred = pred.squeeze(1)


    return pred.T, order

def compute_feature_importance(avail_feats, feature_names, model, X_valid, i_patient, ):
    feature_indices = list(range(len(model.feature_names)))
    random.shuffle(feature_indices)
    pred_without_feat,_ = apply_model_feat_subset([elem for elem in avail_feats if elem not in feature_names], model, X_valid, i_patient, feat_order = feature_indices)
    pred_with_feat,_ = apply_model_feat_subset(avail_feats, model, X_valid, i_patient, feat_order = feature_indices)

    return pred_without_feat[...,-1], pred_with_feat[...,-1]

max_v = 0

for node_idx in range(1, len(node_data_valid)+1):
#for node_idx in [3]:

    types = ["local", "fl"]
    fig, axes = plt.subplots(1, len(types), figsize=(15, 10), sharey=True)

    # Loop over each type and corresponding subplot axis
    for ax, t in zip(axes, types):
        model = models[t][node_idx]
        node_features = node_data_valid[node_idx - 1][2]

        # Compute feature importances
        f_importances = {}
        for f in node_features:
            without_f, with_f = compute_feature_importance(node_features, [f], model, X_valid, i_patient=None)
            f_importances[f] = abs(without_f - with_f).mean(axis=1)

        # Build DataFrame: rows = features, columns = targets
        df = pd.DataFrame(f_importances).T
        df.columns = model.target_names

        # Plot heatmap on the current axis
        im = ax.imshow(df.values, aspect='auto', cmap='Blues', vmin = 0, vmax = 0.3)
        if max(df.values.flatten()) > max_v:
            max_v = max(df.values.flatten()) 
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=90, ha='right')
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_title(f"{t.upper()}")

        # # Annotate each cell
        # for i in range(df.shape[0]):
        #     for j in range(df.shape[1]):
        #         val = df.values[i, j]
        #         ax.text(
        #             j, i, 
        #             f"{val:.2f}",                # format the number
        #             ha="center", 
        #             va="center",
        #             color="black",
        #             fontsize=8
        #         )

    # Add a single colorbar for both subplots
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', location = 'right' )
    cbar.set_label('Importance')

    #plt.tight_layout(rect=[0, 0, 0.95, 1])


    # Save the combined figure
    outfile = f"tmp/{node_data_valid[node_idx - 1][7]}_comparison.png"
    plt.savefig(outfile)

    # for type in ["local", "fl"]:
    #     model = models[type][node_idx]
    #     node_features = node_data_valid[node_idx-1][2]
        

    #     f_importances = {}
    #     for f in node_features:
    #         without_f, with_f = compute_feature_importance(node_features, [f], model, X_valid, i_patient= None)
    #         f_importances[f] = abs(without_f-with_f).mean(axis=1)

    #     # Build DataFrame: rows = targets, columns = methods
    #     df = pd.DataFrame(
    #         f_importances
    #     ).T
    #     df.columns = model.target_names
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(df.values, aspect='auto',cmap ='Blues' )
    #     plt.colorbar()
    #     plt.xticks(range(len(df.columns)), df.columns, rotation = 90, ha='right')
    #     plt.yticks(range(len(df.index)), df.index)

    #     plt.tight_layout()

    #     plt.savefig(f'tmp/{node_data_valid[node_idx-1][7]}_{type}.png')

print(max_v)
print("End of script")
