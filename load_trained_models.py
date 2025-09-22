import torch
import wandb
import os
from modules_modn import ModularModel
from modules_baseline import BaselineModel
 
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


load_dotenv()
os.environ["WANDB_INIT_TIMEOUT"] = "600"
random.seed(42)

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)

api = wandb.Api()

# Instead of getting the latest run artifact via a run,
# load the artifact directly from the project.

project_path = "cctrotte/flmodn"

# Retrieve all runs for your project
runs = api.runs(project_path)
finished_runs = [run for run in runs if run.state == "finished"]

# Sort runs by creation time (most recent first)
latest_run = sorted(finished_runs, key=lambda run: run.created_at, reverse=True)[1]

# Define the run path from the URL
run_path = "cctrotte/flmodn/oo53sm13"
#run_path= "cctrotte/flmodn/78dvtvj2"
#run_path = "cctrotte/flmodn/b9m653f2"

# Load the run
latest_run = wandb.Api().run(run_path)

artifacts = latest_run.logged_artifacts()
best_model = [artifact for artifact in artifacts if artifact.type == 'model'][0]
data_dict = [artifact for artifact in artifacts if artifact.type == 'dataset'][0]
model_dir = best_model.download()
data_dir = data_dict.download()

# load original df
df = pd.read_csv("./datasets/epoct_plus_cleaned.csv")
feature_types = load_config(config_file="epoct_feature_type.yaml")
target_types = load_config(config_file="epoct_target_type.yaml")
df_num = get_numeric_data(df, feature_types, target_types)


with open(data_dir + "/data_artifact.pkl", "rb") as f:
    data_dict = pickle.load(f) 


node_data_valid = data_dict["node_data_valid"]  
node_data_test = data_dict["node_data_test"]
feature_index_mapping = data_dict["feature_index_mapping"]
target_index_mapping = data_dict["target_index_mapping"]
n_samples = data_dict["n_samples"]
all_features = data_dict["all_features"]
all_targets = data_dict["all_targets"]
node_variables = data_dict["node_variables"]

health_facilities = [elem['health_facility_id'] for k, elem in node_variables.items()]

# Load the checkpoint from the downloaded artifact
checkpoint = torch.load(f"{model_dir}/best_models.pt")

models = {}
# Reconstruct and load models from the checkpoint
type = "fl"
for node_id in checkpoint[type].keys():
    model_config = checkpoint[type][node_id]['config']
    state_dict = checkpoint[type][node_id]['state_dict']
    #model = BaselineModel(**model_config)
    model = ModularModel(**model_config)
    model.load_state_dict(state_dict)
    models[node_id] = model
print(f'{sum([checkpoint["fl"][node_id]["auprc"] > checkpoint["local"][node_id]["auprc"] for node_id in checkpoint["fl"].keys()])/len(checkpoint["fl"])}')


# apply some models to the valid/test data
# node_data structure is as follows a list of tuples containing
# (X_node, y_node, feature, target_names, X_node_nan_mask, y_node_nan_mask)

# select a node
node_idx = 1
facility_id = health_facilities[node_idx-1]
X_full = df_num[df_num["health_facility_id"] == facility_id]
model = models[node_idx]
X_valid = node_data_valid[node_idx-1][0]
y_valid = node_data_valid[node_idx-1][1]
X_valid_mask = ~node_data_valid[node_idx-1][4]
model.eval()

indices_valid = node_data_valid[node_idx-1][6]
data_valid = X_full[X_full.patient_id.isin(indices_valid)]
node_features = node_data_valid[node_idx-1][2]

if node_features != model.feature_names:
    raise ValueError(f"Feature names do not match for node {node_idx}.")

node_targets = node_data_valid[node_idx-1][3]
data_valid_features = data_valid[node_features]
data_valid_targets = data_valid[node_targets]

# plot heatmap for test patient
# plot heatmap of predicted probabilities
i_patient = 17#5 # 6,7,8,9 #[10, 12, 17, 18, 23, 25, 26, 35]
gt = y_valid[i_patient]
mask_ = X_valid_mask[i_patient,:]
avail_feats = [feature_name for feature_name in node_features if mask_[model.input_dims[feature_name]].all().item()]
patient_feat_unsc = data_valid_features.iloc[i_patient]

# implement method to select features we want to use from available features, maybe compare x predictions with random shuffling (hopefully all quite similar), compare with x predictions with feature (hopefully all quite sim)
# compute difference between predictions with and without feature
# implement method that does it for all features

def apply_model_feat_subset(feats_to_keep, model, X_valid, i_patient, feat_order=None):
    tmp = torch.full(X_valid[i_patient].shape, False, dtype=torch.bool)
    idx_to_keep = [model.input_dims[f] for f in feats_to_keep]
    idx_to_keep = [i for s in idx_to_keep for i in s]
    tmp[idx_to_keep] = True
    input_ = X_valid[i_patient,:].clone()
    input_[~tmp] = np.nan
    _, pred, order = model(input_.reshape(1,-1), feat_order = feat_order)
    pred = torch.sigmoid(pred).detach().numpy()

    return pred.squeeze(1).T, order

def compute_feature_importance(avail_feats, feature_names, model, X_valid, i_patient, ):
    feature_indices = list(range(len(model.feature_names)))
    random.shuffle(feature_indices)
    pred_without_feat,_ = apply_model_feat_subset([elem for elem in avail_feats if elem not in feature_names], model, X_valid, i_patient, feat_order = feature_indices)
    pred_with_feat,_ = apply_model_feat_subset(avail_feats, model, X_valid, i_patient, feat_order = feature_indices)

    return pred_without_feat[:,-1], pred_with_feat[:,-1]

feat_order = list(range(len(model.feature_names)))
random.shuffle(feat_order)
pred, order = apply_model_feat_subset(avail_feats, model, X_valid, i_patient, feat_order)
plot_heatmap(pred, gt, all_targets, avail_feats, node_features, order, patient_feat_unsc, model, 'heatmap5.png')


feature_names = ["malaria_rdt"]
i1, i2 = compute_feature_importance(avail_feats, feature_names, model, X_valid, i_patient)

# feats_to_keep =  [elem for elem in avail_feats if elem not in feature_names] #'cc_eye', 'cc_feeding'
# tmp = torch.full(X_valid[i_patient].shape, False, dtype=torch.bool)
# idx_to_keep = [model.input_dims[f] for f in feats_to_keep]
# idx_to_keep = [i for s in idx_to_keep for i in s]
# tmp[idx_to_keep] = True
# input_ = X_valid[i_patient,:].clone()
# input_[~tmp] = np.nan
#feature_indices = list(range(len(model.feature_names)))
# randomly shuffle order in which encoders are applied

#random.shuffle(feature_indices)
# _, pred, feat_order = model(input_.reshape(1,-1), feat_order = feature_indices)
# pred = torch.sigmoid(pred).detach().numpy()


print("End of script")
