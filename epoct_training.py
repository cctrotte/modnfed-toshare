import pandas as pd
from utils.utils_epoct import get_numeric_data, get_data_for_nodes, get_class_weights
import torch
import torch.nn as nn
import random
import numpy as np
from modules_modn import ModularModel
from utils.other import *
from utils.logs_wandb import log_wandb_epoct
from modules_baseline import BaselineModel
from utils.training import train_on_node, aggregate_models
from utils.plot_and_save import *
import pickle
from dotenv import load_dotenv
import os
import wandb
from sklearn.model_selection import KFold



if __name__ == "__main__":

    config = load_config(config_file="config_epoct_more.yaml")

    with open("node_splits.pkl", "rb") as f:
        node_splits = pickle.load(f)

    print(config["agg_weight"])


    load_dotenv()
    os.environ["WANDB_INIT_TIMEOUT"] = "600"
    os.environ["WANDB_MODE"] = "offline"


    wandb_api_key = os.getenv("WANDB_API_KEY")
    # Initialize OpenAI client
    wandb.login(key=wandb_api_key)

    base_dir = "saved_models/" + config["foldername"]
    os.makedirs( base_dir, exist_ok=True)

    df = pd.read_csv("./datasets/epoct_plus_cleaned.csv")
    config = load_config(config_file="config_epoct_more.yaml")

    feature_types = load_config(config_file="epoct_feature_type.yaml")
    target_types = load_config(config_file="epoct_target_type.yaml")

    model_type = config["model_type"]
    seed = config["seed"]
    local_epochs = config["local_epochs"]
    federated_rounds = config["federated_rounds"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    state_dim = config["state_dim"]
    feature_decoding = config["feature_decoding"]
    predict_all = config["predict_all"]
    shuffle = config["shuffle"]
    node_variables = config["node_variables"]
    agg_weight = config["agg_weight"]
    weight_ce_loss = config["weight_ce_loss"]
    mu = config["mu"]

    all_ = config["all_"]  # whether to train on almost all nodes

    hidden_layers_enc = config["hidden_layers_enc_shared"]
    hidden_layers_enc_private = config["hidden_layers_enc_private"]
    hidden_layers_dec = config["hidden_layers_dec_shared"]
    hidden_layers_dec_private = config["hidden_layers_dec_private"]
    hidden_layers_decoding = config["hidden_layers_decoding"]

    # map binary variables to 0/1 (other variables not modified)
    df_num = get_numeric_data(df, feature_types, target_types)

    random.seed(seed)

    if all_:
        features_ = [
            "prevention_and_screening",
            "age_c",
            "sex_c",
            "s_abdopain_c",
            "s_bloodstool_c",
            "s_cough_c",
            "s_diarrhea_c",
            "s_fever_c",
            "s_vomit_c",
            "wfa_c",
            "s_runny_nose",
            "stridor_or_grunting",
            "heart_rate",
            "resp_rate",
            "neck_stiffness",
            "grunting_chest_indrawing_difficulty_breathing",
            "muac",
            "pallor",
            "cc_fever",
            "cc_respiratory",
            "cc_eye",
            "cc_ear_throat_mouth",
            "cc_skin_hair",
            "cc_gastro_intestinal",
            "cc_trauma",
            "cc_feeding",
            "crp",
            "hemoglobin",
            "sao2",
            "malaria_rdt_or_microscopy",
            "chest_indrawing",
            "stridor_grunting_difficulty_breathing",
            "rr_age_cat",
        ]
        features_small = [
            "prevention_and_screening",
            "age_c",
            "sex_c",
            "s_abdopain_c",
            "s_bloodstool_c",
            "s_cough_c",
            "s_diarrhea_c",
            "s_fever_c",
            "s_vomit_c",
            "wfa_c",
            "s_runny_nose",
            "stridor_or_grunting",
            "heart_rate",
            "resp_rate",
            "neck_stiffness",
            "grunting_chest_indrawing_difficulty_breathing",
            "muac",
            "pallor",
            "hemoglobin",
            "sao2",
            "malaria_rdt_or_microscopy",
        ]
        targets_ = [
            "dxfinal_anemia",
            "dxfinal_diarrhea",
            "dxfinal_fws",
            "dxfinal_malaria",
            "dxfinal_malnut",
            "dxfinal_urti",
            "dxfinal_bact_pna",
            "dxfinal_viral_pna_or_common_cold",
        ]
        targets_small = [
            "dxfinal_bact_pna",
            "dxfinal_malaria",
            "dxfinal_anemia",
        ]
        count_ = df_num["health_facility_id"].value_counts()
        ids_ = list(count_[count_ > 10].index)


        # 30 with little data
        ids_ = [267, 299, 297, 333, 292, 336, 335, 324, 344, 329, 332, 288, 331, 340,
       337, 327, 315, 257, 343, 318, 281, 303, 339, 342, 280, 290, 338, 311,
       330, 270]

        node_variables = {"node_" + str(i): {"health_facility_id": i, "features": sorted(features_), "targets": sorted(targets_)} for i in sorted(ids_)}

    else:

        # ensure that node, features and targets are in consistent order
        node_variables = {
            key: {
                "health_facility_id": node_variables[key]["health_facility_id"],
                "features": sorted(node_variables[key]["features"]),
                "targets": sorted(node_variables[key]["targets"]),
            }
            for key in sorted(node_variables.keys())
        }
        ids_ = sorted(list([node_variables[key]["health_facility_id"] for key in node_variables.keys()]))
    config["node_variables"] = node_variables
    wandb.init(project="flmodn", config=config)
    artifact = wandb.Artifact('best_model_checkpoint', type='model')

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
    # save the data 
    data_artifact = wandb.Artifact("data_artifact", type="dataset")

    data_dict = {
    "X_centralized_train": X_centralized_train,
    "X_centralized_valid": X_centralized_valid,
    "y_centralized_train": y_centralized_train,
    "y_centralized_valid": y_centralized_valid,
    "node_data_train": node_data_train,
    "node_data_valid": node_data_valid,
    "node_data_test": node_data_test,
    "feature_index_mapping": feature_index_mapping,
    "target_index_mapping": target_index_mapping,
    "n_samples": n_samples,
    "all_features": all_features,
    "all_targets": all_targets,
    "node_variables": node_variables,}
    with open("data_artifact.pkl", "wb") as f:
        pickle.dump(data_dict, f)   
    data_artifact.add_file("data_artifact.pkl")
    wandb.log_artifact(data_artifact)
    # compute class weights
    # todo adapt in case we have continuous targets
    target_weights = get_class_weights(
        weight_ce_loss,
        y_centralized_train,
        node_data_train,
        all_targets,
        list(node_variables.keys()),
    )
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)

    if model_type == "modn":

        # shared FL model
        global_config = {"feature_names":all_features,
            "target_names_and_types":{name: "binary" for name in all_targets},
            "state_dim":state_dim,
            "hidden_layers_enc":hidden_layers_enc,
            "hidden_layers_enc_private":hidden_layers_enc_private,
            "hidden_layers_dec":hidden_layers_dec,
            "hidden_layers_dec_private":hidden_layers_dec_private,
            "hidden_layers_feat_dec":hidden_layers_decoding,
            "feat_decoding":feature_decoding,
            "predict_all":predict_all,
            "shuffle":shuffle,
            "input_dims":feature_index_mapping,
            "output_dims": target_index_mapping,}
        global_model = ModularModel(
            **global_config
        )
        # centralized model trained on concatenated databases (used as upper baseline)
        centralized_config = {key: elem for key,elem in global_config.items() if key not in ["hidden_layers_enc_private","hidden_layers_dec_private"]}  
        centralized_config["hidden_layers_enc"] = hidden_layers_enc + hidden_layers_enc_private
        centralized_config["hidden_layers_dec"] = hidden_layers_dec + hidden_layers_dec_private
        centralized_model = ModularModel(
        **centralized_config
        )

    else:
        # baseline model
        global_config = {"input_dim": sum([len(elem) for elem in feature_index_mapping.values()]),
                         "feature_names":all_features,
            "state_dim": state_dim,
            "hidden_layers_enc": hidden_layers_enc_private + hidden_layers_enc,
            "hidden_layers_dec": hidden_layers_dec + hidden_layers_dec_private,
            "hidden_layers_decoding": hidden_layers_decoding,
            "feat_decoding": feature_decoding,
            "output_dims": target_index_mapping,
            "target_names_and_types": {name: "binary" for name in all_targets},}
        global_model = BaselineModel(
            **global_config
        )

        centralized_config = {key: elem for key,elem in global_config.items()}
        centralized_model = BaselineModel(
            **centralized_config
        )

    print(f"Number of model parameters {count_parameters(global_model)}")

    # for feature decoding, not necessarily used
    criterion_features = nn.MSELoss()

    # criterion_target = nn.BCEWithLogitsLoss() if output_type == "binary" else nn.MSELoss()
    # hard coded for now, modify if we have some continuous target
    criterion_targets = {
        node: {
            name: nn.BCEWithLogitsLoss(pos_weight=target_weights[node][name])
            for name in all_targets
        }
        for node in node_variables.keys()
    }
    criterion_targets["centralized"] = {
        name: nn.BCEWithLogitsLoss(pos_weight=target_weights["centralized"][name])
        for name in all_targets
    }
    local_models = []
    local_configs = []
    local_models_independent = []
    local_configs_independent = []
    if model_type == "modn":
        # local models trained with FL. Perform a few local training steps before aggregation of parameters with global model
        

        for _, _, features, targets, _, _,_, node_id in node_data_train: 
            local_config = {key: elem for key,elem in global_config.items() if key not in ["feature_names", "target_names_and_types"]}
            local_config["node_id"]= node_id
            local_config["feature_names"] = features
            local_config["target_names_and_types"] = {name: "binary" for name in targets}
            local_models.append(ModularModel(**local_config))
            local_configs.append(local_config)

            local_config_independent = {key: elem for key,elem in centralized_config.items() if key not in ["feature_names", "target_names_and_types",]}
            local_config_independent["node_id"] = node_id
            local_config_independent["feature_names"] = features
            local_config_independent["target_names_and_types"] = {name: "binary" for name in targets}
            local_models_independent.append(ModularModel(**local_config_independent))
            local_configs_independent.append(local_config_independent)

        

    else:
        # baseline
        for _, _, features, targets, _, _,_, node_id in node_data_train: 

            local_config = {key:elem for key,elem in global_config.items() if key not in ["feature_names"]}
            local_config["node_id"]= node_id

            local_config["feature_names"] = features
            local_models.append(BaselineModel(**local_config))
            local_configs.append(local_config)
            local_config_independent = {key:elem for key,elem in global_config.items()}
            local_config_independent["node_id"] = node_id

            local_models_independent.append(BaselineModel(**local_config_independent))
            local_configs_independent.append(local_config_independent)

    for i, local_model in enumerate(local_models):
        #local_model_ind.load_state_dict(global_model.state_dict(), strict=False)
        local_model.load_state_dict(global_model.state_dict(), strict=False)

    # Define loss types and corresponding datasets
    loss_types = ["fl", "local", "centralized"]
    data_dicts = [node_data_train, node_data_valid, node_data_test]

    # Initialize dict for saving losses
    losses = {
        split: init_all_loss_dicts(loss_types, data_dicts)
        for split in ["train", "valid", "test"]
    }
    f1_scores = {
        split: init_all_loss_dicts(loss_types, data_dicts)
        for split in ["train", "valid", "test"]
    }
    auprc_scores = {
        split: init_all_loss_dicts(loss_types, data_dicts)
        for split in ["train", "valid", "test"]
    }
    auprc_scores_neg = {
        split: init_all_loss_dicts(loss_types, data_dicts)
        for split in ["train", "valid", "test"]
    }
    auc_scores = {
        split: init_all_loss_dicts(loss_types, data_dicts)
        for split in ["train", "valid", "test"]
    }

    avail_perc = {node_id: {f: 1- (node_data_train[node_id-1][4][:,feature_index_mapping[f]].all(dim=1).sum()/len(node_data_train[node_id-1][4])).item() for f in all_features} for node_id in range(1, len(node_data_train)+1)}

    # keep track of the best models
    current_best_valid_auprc = {"fl": {node_id: 0 for node_id in range(1, len(node_data_train)+1)}, "local": {node_id:0 for node_id in range(1, len(node_data_train)+1)}, "centralized": {node_id:0 for node_id in range(1, len(node_data_train)+1)}}
    best_models = {"fl": {node_id: {"config": {}, "state_dict": {}, "auprc": 0, "epoch": 0} for node_id in range(1, len(node_data_train)+1)}, "local":  {node_id: {"config": {}, "state_dict": {}, "auprc": 0, "epoch": 0}  for node_id in range(1, len(node_data_train)+1)}, "centralized":  {node_id: {"config": {}, "state_dict": {}, "auprc": 0, "epoch": 0} for node_id in range(1, len(node_data_train)+1)}}

    feature_weights = {node_id: {f: 1 for f in all_features} for node_id in range(1, len(node_data_train)+1)}
    # Training federated model and local models
    for round_num in range(federated_rounds):
        print(f"--- Federated Round {round_num + 1} ---")

        # Federated learning: Train each node and aggregate
        for i_model, (
            node,
            (X, y, feature_names, target_names, nan_mask, nan_target_mask,_, _),
        ) in enumerate(zip(node_variables.keys(), node_data_train)):
            local_model = local_models[i_model]

            # Check: Findout if some local modules dont exist in global
            unmatched_local_keys = set(local_model.state_dict().keys()) - set(
                global_model.state_dict().keys()
            )

            # Print warning if some local parameters won't be updated
            if unmatched_local_keys:
                print(
                    "WARNING: The following local model parameters will NOT be updated:"
                )
                for key in unmatched_local_keys:
                    print(f"   - {key}")
            # # load shared parameters from global model
            # local_model.load_state_dict(global_model.state_dict(), strict=False)

            global_dict = global_model.state_dict()
            # Filter out private params
            shared_global_dict = {
                k: v for k, v in global_dict.items() if "private" not in k
            }
            local_dict = local_model.state_dict()
            local_dict.update(shared_global_dict)
            local_model.load_state_dict(local_dict, strict=False)

            # local training
            train_on_node(
                local_model,
                global_model,
                X,
                y,
                criterion_features,
                criterion_targets[node],
                local_epochs,
                feature_names,
                target_names,
                nan_mask,
                nan_target_mask,
                batch_size,
                learning_rate,
                feature_decoding,
                mu = mu,
            )


        # Aggregate the parameters of the local models to update the global model
        aggregate_models(
            global_model,
            local_models,
            n_samples,
            weight=agg_weight,
            model_type=model_type,
        )


        # Local-only training: Train each independent model (lower baseline) on local data only
        for (
            local_model_ind,
            node,
            (X, y, feature_names, target_names, nan_mask, nan_target_mask,_, _),
        ) in zip(local_models_independent, node_variables.keys(), node_data_train):
            train_on_node(
                local_model_ind,
                global_model,
                X,
                y,
                criterion_features,
                criterion_targets[node],
                local_epochs,
                feature_names,
                target_names,
                nan_mask,
                nan_target_mask,
                batch_size,
                learning_rate,
                feature_decoding,
            )
        # centralized training on all data (upper baseline) on concatenated data

        train_on_node(
            centralized_model,
            global_model,
            X_centralized_train,
            y_centralized_train,
            criterion_features,
            criterion_targets["centralized"],
            local_epochs,
            all_features,
            all_targets,
            torch.isnan(X_centralized_train),
            torch.isnan(y_centralized_train),
            batch_size,
            learning_rate,
            feature_decoding,
        )

        # Evaluate models
        global_model.eval()
        feature_weights = {}
        with torch.no_grad():
            for node_id, (
                node,
                (X, y, feature_names, target_names, nan_mask, nan_target_mask,_, _),
            ) in enumerate(zip(node_variables.keys(), node_data_train), 1):
                


                local_model = local_models[node_id - 1]
                # local_model = local_models[0]

                local_model.eval()

                global_dict = global_model.state_dict()
                # Filter out private params
                shared_global_dict = {
                    k: v for k, v in global_dict.items() if "private" not in k
                }
                local_dict = local_model.state_dict()
                local_dict.update(shared_global_dict)
                local_model.load_state_dict(local_dict, strict=False)

                X_test, y_test, _, _, nan_mask_test, nan_target_mask_test,_,_ = (
                    node_data_test[node_id - 1]
                )
                X_valid, y_valid, _, _, nan_mask_valid, nan_target_mask_valid,_,_ = (
                    node_data_valid[node_id - 1]
                )

                data = [
                    (X, y, nan_mask, nan_target_mask),  # Training Data
                    (
                        X_valid,
                        y_valid,
                        nan_mask_valid,
                        nan_target_mask_valid,
                    ),  # Validation Data
                    (X_test, y_test, nan_mask_test, nan_target_mask_test),  # Test Data
                ]
                _, pred_valid, order = local_model(X_valid)
                pred_valid = torch.sigmoid(pred_valid)
                y_valid_weights = y_valid.clone()
                y_valid_weights[y_valid_weights == 0] = -1
               
                plot_auprc = False

                losses, f1_scores, auprc_scores, auprc_scores_neg, auc_scores = (
                    compute_and_store_losses(
                        local_model,
                        "fl",
                        data,
                        losses,
                        f1_scores,
                        auprc_scores,
                        auprc_scores_neg,
                        auc_scores,
                        node_id,
                        feature_names,
                        target_names,
                        criterion_targets[node],
                        plot_auprc=plot_auprc,
                    )
                )
                losses, f1_scores, auprc_scores, auprc_scores_neg, auc_scores = (
                    compute_and_store_losses(
                        local_models_independent[node_id - 1],
                        "local",
                        data,
                        losses,
                        f1_scores,
                        auprc_scores,
                        auprc_scores_neg,
                        auc_scores,
                        node_id,
                        feature_names,
                        target_names,
                        criterion_targets[node],
                        plot_auprc=plot_auprc,
                    )
                )
                losses, f1_scores, auprc_scores, auprc_scores_neg, auc_scores = (
                    compute_and_store_losses(
                        centralized_model,
                        "centralized",
                        data,
                        losses,
                        f1_scores,
                        auprc_scores,
                        auprc_scores_neg,
                        auc_scores,
                        node_id,
                        feature_names,
                        target_names,
                        criterion_targets["centralized"],
                        plot_auprc=plot_auprc,
                    )
                )

                # Print debugging information
                print_debugging_losses(losses, "fl", node_id)
                print_debugging_losses(losses, "local", node_id)
                print_debugging_losses(losses, "centralized", node_id)

                # keep track of the best model
                current_auprc_fl = np.mean([auprc_scores["valid"]["fl"][node_id][t][-1] for t in target_names])
                current_auprc_local = np.mean([auprc_scores["valid"]["local"][node_id][t][-1] for t in target_names])
                current_auprc_centralized = np.mean([auprc_scores["valid"]["centralized"][node_id][t][-1] for t in target_names])

                current_f1_fl = np.mean([f1_scores["valid"]["fl"][node_id][t][-1] for t in target_names])
                current_f1_local = np.mean([f1_scores["valid"]["local"][node_id][t][-1] for t in target_names]) 
                current_f1_centralized = np.mean([f1_scores["valid"]["centralized"][node_id][t][-1] for t in target_names])

                
                if current_auprc_fl > current_best_valid_auprc["fl"][node_id]:
                    best_models["fl"][node_id]["state_dict"] = local_model.state_dict()
                    best_models["fl"][node_id]["config"] = local_configs[node_id - 1]
                    best_models["fl"][node_id]["auprc"] = current_auprc_fl
                    best_models["fl"][node_id]["f1"] = current_f1_fl
                    best_models["fl"][node_id]["epoch"] = round_num
                    current_best_valid_auprc["fl"][node_id] = current_auprc_fl
                if current_auprc_local > current_best_valid_auprc["local"][node_id]:
                    best_models["local"][node_id]["state_dict"] = local_models_independent[node_id - 1].state_dict()
                    best_models["local"][node_id]["config"] = local_configs_independent[node_id - 1]
                    best_models["local"][node_id]["auprc"] = current_auprc_local
                    best_models["local"][node_id]["f1"] = current_f1_local
                    best_models["local"][node_id]["epoch"] = round_num
                    current_best_valid_auprc["local"][node_id] = current_auprc_local
                if current_auprc_centralized > current_best_valid_auprc["centralized"][node_id]:
                    best_models["centralized"][node_id]["state_dict"] = centralized_model.state_dict()
                    best_models["centralized"][node_id]["config"] = centralized_config
                    best_models["centralized"][node_id]["auprc"] = current_auprc_centralized
                    best_models["centralized"][node_id]["f1"] = current_f1_centralized
                    best_models["centralized"][node_id]["epoch"] = round_num
                    current_best_valid_auprc["centralized"][node_id] = current_auprc_centralized

            
            log_wandb_epoct(losses, node_variables, all_targets, f1_scores, auprc_scores, auprc_scores_neg, auc_scores, round_num)

    

    # some plots and save results
    bsl_prediction = plot_losses_per_node(
        losses, node_data_train, node_data_test, model_type
    )
    plot_f1_scores_per_node(f1_scores, node_data_train, model_type)

    plot_f1_scores_per_node_and_target(
        all_targets, f1_scores, node_data_train, model_type
    )
    plot_losses_per_node_and_target(
        all_targets, losses, node_data_train, node_data_test, model_type
    )

    wandb.log({"fl_auprc_best": np.mean([best_models["fl"][node_id]["auprc"] for node_id in range(1, len(node_data_train)+1)]),
               "local_auprc_best": np.mean([best_models["local"][node_id]["auprc"] for node_id in range(1, len(node_data_train)+1)]),
               "best_local_epochs": [best_models["local"][node_id]["epoch"] for node_id in range(1, len(node_data_train)+1)],
               "best_fl_epochs": [best_models["fl"][node_id]["epoch"] for node_id in range(1, len(node_data_train)+1)],
               "centralized_auprc_best": np.mean([best_models["centralized"][node_id]["auprc"] for node_id in range(1, len(node_data_train)+1)]),
               "fl_f1_best": np.mean([best_models["fl"][node_id]["f1"] for node_id in range(1, len(node_data_train)+1)]),
               "local_f1_best": np.mean([best_models["local"][node_id]["f1"] for node_id in range(1, len(node_data_train)+1)]),
               "centralized_f1_best": np.mean([best_models["centralized"][node_id]["f1"] for node_id in range(1, len(node_data_train)+1)]),
               "feature_weights": feature_weights,
               "FL_better_than_local_f1": sum([best_models["fl"][node_id]["f1"] > best_models["local"][node_id]["f1"] for node_id in range(1, len(node_data_train)+1)])/len(node_data_train),
               "FL_better_than_local_auprc": sum([best_models["fl"][node_id]["auprc"] > best_models["local"][node_id]["auprc"] for node_id in range(1, len(node_data_train)+1)])/len(node_data_train),})
    # print(f'{sum([checkpoint["fl"][node_id]["auprc"] > checkpoint["local"][node_id]["auprc"] for node_id in checkpoint["fl"].keys()])/len(checkpoint["fl"])}')



    print(f"Number of model parameters {count_parameters(global_model)}")
    avg_losses, std_losses = compute_avg_std(losses)
    plot_avg_losses(avg_losses, std_losses, federated_rounds, model_type)





    # # save best models
    # save_path = os.path.join(base_dir, "best_models.pt")

    # torch.save(best_models, save_path)


    print("End of script")
