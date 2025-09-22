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
import glob
from sklearn.model_selection import KFold

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

model_dir = "cv_models/final30small_5f/saved_bsl_bis/"
#model_dir = "cv_models/final30small_bis/saved_bsl/"

save_path = 'cv_models_perf/' + model_dir.split('/')[-2] + '/'
os.makedirs(save_path, exist_ok=True)


pt_paths = glob.glob(os.path.join(model_dir, "*.pt"))

checkpoints = {}

# 4. loop through and load
for path in pt_paths:
    # extract a friendly name (e.g. the filename without extension)
    name = int(path.split('__')[-1].split('.')[0])
    # pick whichever matches how you saved:
    model = torch.load(path, map_location="cpu")

    # put it in your dict
    checkpoints[name] = model

df = pd.read_csv("./datasets/epoct_plus_cleaned.csv")
df_num = get_numeric_data(df, feature_types, target_types)

model_type = 'bsl'
all = True

hold_out = False # whether to evaluate on test data from training nodes or on nodes not used for traiin
types = ["fl", "centralized"] if hold_out else ["fl", "local", "centralized"]
models = {fold: {type:{} for type in types} for fold in checkpoints.keys()}
for fold in models.keys():
    for type in types:
        for node_id in checkpoints[fold][type].keys():
            model_config = checkpoints[fold][type][node_id]['config']
            state_dict = checkpoints[fold][type][node_id]['state_dict']
            if model_type == 'modn':
                model = ModularModel(**model_config)
            else:
                model = BaselineModel(**model_config)
            model.load_state_dict(state_dict)
            models[fold][type][node_id] = model

train_ids_ = sorted([models[0]['fl'][k].node_id for k in models[0]['fl'].keys() if k!= 'global'])

count_ = df_num["health_facility_id"].value_counts()
all_ids_ = list(count_[(count_ > 10) & (count_<=1000)].index)
holdout_ids_ = [elem for elem in all_ids_ if elem not in train_ids_] #train_ids_

node_variables_train = {"node_" + str(i): {"health_facility_id": models[0]["fl"][ix+1].node_id, "features": sorted(models[0]["fl"][ix+1].feature_names), "targets": sorted(list(models[0]['fl'][1].target_names_and_types.keys()))} for ix, i in enumerate(sorted(train_ids_))}

#health_facilities_train = [elem['health_facility_id'] for k, elem in node_variables_train.items()]

all_features = sorted(
    list(set([f for elem in node_variables_train.values() for f in elem["features"]]))
)
all_targets = sorted(
    list(set([f for elem in node_variables_train.values() for f in elem["targets"]]))
)


node_variables_hold_out = {"node_" + str(o_i): {"health_facility_id": o_i, "features": sorted(all_features), "targets": sorted(all_targets)} for o_i in sorted(holdout_ids_)}


kf_train = KFold(n_splits=5, shuffle=False,)
splits_per_node_train = {
n: list(kf_train.split(np.array(node_splits[n]["train"])))
    for n in train_ids_}

kf_holdout = KFold(n_splits=5, shuffle=False,)
splits_per_node_holdout = {
n: list(kf_holdout.split(np.array(node_splits[n]["train"])))
    for n in holdout_ids_}

node_splits_fold = {}

auprc_all = {t: [] for t in types}
f1_all = {t: [] for t in types}

if model_type == "modn":
    auprc_all_ensemble = {t: [] for t in types}
    f1_all_ensemble = {t: [] for t in types}

fl_better_all = []

ids_ = holdout_ids_ if hold_out else train_ids_
splits_per_node = splits_per_node_holdout if hold_out else splits_per_node_train 
node_variables = node_variables_hold_out if hold_out else node_variables_train
for fold in models.keys():

    all_targets_tmp = []
    all_preds_tmp = {t: [] for t in types}
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
    print(f"------{fold}")


    aus = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}
    f1s = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}

    if model_type == "modn":
        aus_ensemble = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}
        f1s_ensemble = {type: {target_name: [] for target_name in sorted(all_targets)} for type in types}

    fl_better = []


    for node_idx in range(1, len(node_data_valid)+1):

        facility_id = node_data_valid[node_idx-1][7]
        X_full = df_num[df_num["health_facility_id"] == facility_id]
        model_fl = models[fold]["fl"]["global"] if hold_out else models[fold]["fl"][node_idx]
        if not hold_out:
            model_local = models[fold]["local"][node_idx]
            model_local.eval()

        model_cl = models[fold]["centralized"]["global"] if hold_out else models[fold]["centralized"][node_idx]
        # X_test = node_data_valid[node_idx-1][0]
        # y_test = node_data_valid[node_idx-1][1]
        # X_test_mask = ~node_data_valid[node_idx-1][4]
        # y_test_mask = ~node_data_valid[node_idx-1][5]
        X_test = node_data_test[node_idx-1][0]
        y_test = node_data_test[node_idx-1][1]
        X_test_mask = ~node_data_test[node_idx-1][4]
        y_test_mask = ~node_data_test[node_idx-1][5]
        model_fl.eval()
        model_cl.eval()

        all_targets_tmp.append(y_test)


        # compute some performance metrics
        preds_fl = torch.sigmoid(model_fl(X_test)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_fl(X_test)[1]).detach() 
        preds_fl_class = (preds_fl >= 0.5).float()

        if not hold_out:

            preds_local =  torch.sigmoid(model_local(X_test)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_local(X_test)[1]).detach() 
            preds_local_class = (preds_local >= 0.5).float()

        preds_cl = torch.sigmoid(model_cl(X_test)[1][-1,:,:]).detach() if (model_type == "modn" and all) else  torch.sigmoid(model_cl(X_test)[1]).detach()
        preds_cl_class = (preds_cl >= 0.5).float()

        fl_better_than_local = 0
        num_tars = 0

        # ensembling
        if model_type == "modn":
            preds_fl_ensemble = []
            preds_local_ensemble = []
            preds_cl_ensemble = []
            for i in range(10):
                preds_fl_ensemble.append(torch.sigmoid(model_fl(X_test, )[1][-1,:,:]).detach())
                preds_cl_ensemble.append(torch.sigmoid(model_cl(X_test, )[1][-1,:,:]).detach())
                if not hold_out:
                    preds_local_ensemble.append(torch.sigmoid(model_local(X_test,)[1][-1,:,:]).detach())
            
            preds_fl_ensemble = torch.mean(torch.stack(preds_fl_ensemble), dim = 0)
            preds_fl_ensemble_class = (preds_fl_ensemble >= 0.5).float()

            if not hold_out:
                preds_local_ensemble = torch.mean(torch.stack(preds_local_ensemble), dim = 0)
                preds_local_ensemble_class = (preds_local_ensemble >= 0.5).float()

            preds_cl_ensemble = torch.mean(torch.stack(preds_cl_ensemble), dim = 0)
            preds_cl_ensemble_class = (preds_cl_ensemble >= 0.5).float()

            all_preds_tmp["fl"].append(preds_fl_ensemble)
            all_preds_tmp["centralized"].append(preds_cl_ensemble)

        #for i, t in enumerate(sorted(model_fl.target_names)):
        for i, t in enumerate(sorted(list(model_cl.target_names_and_types.keys()))):

            target = y_test[:, model_fl.output_dims[t]]

            if len(np.unique(target)) > 1:
                num_tars += 1
                aus["fl"][t].append(average_precision_score(target, preds_fl[:,i]))
                f1s["fl"][t].append(f1_score(target, preds_fl_class[:,i], average="macro"))

                if model_type == "modn":
                
                    aus_ensemble["fl"][t].append(average_precision_score(target, preds_fl_ensemble[:,i]))
                    f1s_ensemble["fl"][t].append(f1_score(target, preds_fl_ensemble_class[:,i], average="macro"))

                if not hold_out:
                    aus["local"][t].append(average_precision_score(target, preds_local[:,i]))
                    f1s["local"][t].append(f1_score(target, preds_local_class[:,i], average="macro"))

                    if model_type == "modn":
                        aus_ensemble["local"][t].append(average_precision_score(target, preds_local_ensemble[:,i]))
                        f1s_ensemble["local"][t].append(f1_score(target, preds_local_ensemble_class[:,i], average="macro"))
                    
                    if average_precision_score(target, preds_fl[:,i]) >= average_precision_score(target, preds_local[:,i]):
                        fl_better_than_local += 1
        
        
        if not hold_out:
            fl_better.append(fl_better_than_local/num_tars)

        #for i,t in enumerate(sorted(model_cl.target_names)):
        for i,t in enumerate(sorted(model_cl.target_names_and_types.keys())):

            if t in model_fl.target_names_and_types.keys():
                target = y_test[:, model_cl.output_dims[t]]

                if len(np.unique(target)) > 1:
                    aus["centralized"][t].append(average_precision_score(target, preds_cl[:,i]))
                    f1s["centralized"][t].append(f1_score(target, preds_cl_class[:,i], average="macro"))

                    if model_type == "modn":
                        aus_ensemble["centralized"][t].append(average_precision_score(target, preds_cl_ensemble[:,i]))
                        f1s_ensemble["centralized"][t].append(f1_score(target, preds_cl_ensemble_class[:,i], average="macro"))
        

    print(fl_better)

    aus_mean = {type: {t: np.mean(aus[type][t]) for t in aus[type]} for type in types}
    f1s_mean = {type: {t: np.mean(f1s[type][t]) for t in f1s[type]} for type in types}

    if model_type == "modn":
        aus_mean_ensemble = {type: {t: np.mean(aus_ensemble[type][t]) for t in aus_ensemble[type]} for type in types}
        f1s_mean_ensemble = {type: {t: np.mean(f1s_ensemble[type][t]) for t in f1s_ensemble[type]} for type in types}
    #
    for t in types:
        auprc_all[t].append(np.mean(list(aus_mean[t].values())))
        f1_all[t].append(np.mean(list(f1s_mean[t].values())))
        if model_type == "modn":
            auprc_all_ensemble[t].append(np.mean(list(aus_mean_ensemble[t].values())))
            f1_all_ensemble[t].append(np.mean(list(f1s_mean_ensemble[t].values())))

    
    print(f'FL: aus mean {np.mean(list(aus_mean["fl"].values()))} f1 mean {np.mean(list(f1s_mean["fl"].values()))}')
    if not hold_out:
        print(f'Local: aus mean {np.mean(list(aus_mean["local"].values()))} f1 mean {np.mean(list(f1s_mean["local"].values()))}')
    print(f'Centralized: aus mean {np.mean(list(aus_mean["centralized"].values()))} f1 mean {np.mean(list(f1s_mean["centralized"].values()))}')

    if model_type == "modn":
        print(f"Ensembling....")
        print(f'FL: aus mean {np.mean(list(aus_mean_ensemble["fl"].values()))} f1 mean {np.mean(list(f1s_mean_ensemble["fl"].values()))}')

        if not hold_out:
            print(f'Local: aus mean {np.mean(list(aus_mean_ensemble["local"].values()))} f1 mean {np.mean(list(f1s_mean_ensemble["local"].values()))}')
        print(f'Centralized: aus mean {np.mean(list(aus_mean_ensemble["centralized"].values()))} f1 mean {np.mean(list(f1s_mean_ensemble["centralized"].values()))}')

    if not hold_out:
        fl_better_all.append(sum([True if elem > 0.5 else False for elem in fl_better])/len(fl_better))


print(f"-------MEAN AND STD ACCROSS FOLDS----------")

print(f'AUPRC Centralized mean: {np.mean(auprc_all["centralized"])} +- {np.std(auprc_all["centralized"])}')
print(f'AUPRC FL mean: {np.mean(auprc_all["fl"])} +- {np.std(auprc_all["fl"])}')

if not hold_out:
    print(f'AUPRC Local mean: {np.mean(auprc_all["local"])} +- {np.std(auprc_all["local"])}')

if not hold_out:
    print(f'FL better: {np.mean(fl_better_all)} +- {np.std(fl_better_all)}')

if model_type == "modn":
    print(f'Ensembling: AUPRC Centralized mean: {np.mean(auprc_all_ensemble["centralized"])} +- {np.std(auprc_all_ensemble["centralized"])}')
    print(f'Ensembling: AUPRC FL mean: {np.mean(auprc_all_ensemble["fl"])} +- {np.std(auprc_all_ensemble["fl"])}')

    if not hold_out:
        print(f'Ensembling: AUPRC Local mean: {np.mean(auprc_all_ensemble["local"])} +- {np.std(auprc_all_ensemble["local"])}')


print(f"-------MEAN AND STD ACCROSS FOLDS----------")

print(f'F1 Centralized mean: {np.mean(f1_all["centralized"])} +- {np.std(f1_all["centralized"])}')
print(f'F1 FL mean: {np.mean(f1_all["fl"])} +- {np.std(f1_all["fl"])}')

if not hold_out:
    print(f'AUPRC Local mean: {np.mean(f1_all["local"])} +- {np.std(f1_all["local"])}')

if model_type == "modn":
    print(f'Ensembling: F1 Centralized mean: {np.mean(f1_all_ensemble["centralized"])} +- {np.std(f1_all_ensemble["centralized"])}')
    print(f'Ensembling: F1 FL mean: {np.mean(f1_all_ensemble["fl"])} +- {np.std(f1_all_ensemble["fl"])}')

    if not hold_out:
        print(f'Ensembling: F1 Local mean: {np.mean(f1_all_ensemble["local"])} +- {np.std(f1_all_ensemble["local"])}')

# save computed metrics
to_save = {'auprc': auprc_all, 'f1': f1_all}
if model_type == 'modn':
    to_save['auprc_ensemble'] = auprc_all_ensemble
    to_save['f1_ensemble'] = f1_all_ensemble

# construct the file path (add .pkl extension)
#file_path = f"{save_path}_hold_out_{hold_out}.pkl"

file_path = f"{save_path}_hold_out_{hold_out}_03.pkl"
#file_path = f"{save_path}_alldata_03.pkl"


# # write the metrics dict to disk
# with open(file_path, 'wb') as f:
#     pickle.dump(to_save, f)    
print("end of script")




