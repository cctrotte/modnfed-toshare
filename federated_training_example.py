# example federated learning script for modn or the mlp baseline on synthetic data

import torch
import torch.nn as nn
import random
import numpy as np
from modules_modn import ModularModel
from utils.other import *
from modules_baseline import BaselineModel
import itertools
from utils.training import train_on_node, aggregate_models
from utils.plot_and_save import *
from utils.data import generate_data_for_nodes
import wandb
import os
from dotenv import load_dotenv


if __name__ == "__main__":

    load_dotenv()
    os.environ["WANDB_MODE"] = "offline"

    wandb_api_key = os.getenv("WANDB_API_KEY")
    # Initialize OpenAI client
    wandb.login(key=wandb_api_key)

    # load config files
    train_config = load_config(config_file="config_training.yaml")
    data_config = load_config(config_file="config_data.yaml")
    output_type = data_config["output_type"]

    keys, values = zip(*train_config.items())
    keys_data, values_data = zip(*data_config.items())

    # all possible features sorted in alphabetical order
    all_features = sorted(
        list(
            set([f for features in data_config["node_feature_sets"] for f in features])
        )
    )

    wandb.init(project="flmodn_synth", config={**train_config, **data_config})

    # generate synthetic data for each node
    (
        X_centralized_train,
        X_centralized_valid,
        y_centralized_train,
        y_centralized_valid,
        node_data_train,
        node_data_valid,
        node_data_test,
    ) = generate_data_for_nodes(
        all_features,
        data_config["node_feature_sets"],
        data_config["n_samples"],
        data_config["impute"],
        output_type,
        data_config["nan_ratio"],
        data_config["data_seed"],
    )

    # iterate over parameter configurations from config file
    for combination in itertools.product(*values):
        # set parameter values
        config_combi = {}
        for key, value in zip(keys, combination):
            config_combi[key] = value
        model_type = config_combi["model_type"]
        seed = config_combi["seed"]
        local_epochs = config_combi["local_epochs"]
        federated_rounds = config_combi["federated_rounds"]
        batch_size = config_combi["batch_size"]
        learning_rate = config_combi["learning_rate"]
        state_dim = config_combi["state_dim"]
        feature_decoding = config_combi["feature_decoding"]
        predict_all = config_combi["predict_all"]
        shuffle = config_combi["shuffle"]

        hidden_layers_enc = config_combi["hidden_layers_enc"]
        hidden_layers_dec = config_combi["hidden_layers_dec"]
        hidden_layers_decoding = config_combi["hidden_layers_decoding"]

        # check compatibility of some arguments
        if model_type == "bsl" and data_config["impute"] == False:
            raise ValueError(
                "Baseline model cannot be trained without imputing missing values"
            )
        if model_type == "modn" and data_config["impute"] == True:
            raise ValueError(
                "MoDN  doesnt need to be trained with imputed missing values"
            )

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)

        if model_type == "modn":
            # hidden_layers = [16, 16]
            # shared FL model
            global_model = ModularModel(
                feature_names=all_features,
                state_dim=state_dim,
                hidden_layers_enc=hidden_layers_enc,
                hidden_layers_dec=hidden_layers_dec,
                hidden_layers_feat_dec=hidden_layers_decoding,
                feat_decoding=feature_decoding,
                # only one target
                target_names_and_types={"t1": output_type},
                predict_all=predict_all,
                shuffle=shuffle,
                input_dims={f: [i] for i, f in enumerate(all_features)},
                output_dims={"t1": 0},
            )
            # centralized model trained on concatenated databases (used as upper baseline)
            centralized_model = ModularModel(
                feature_names=all_features,
                state_dim=state_dim,
                hidden_layers_enc=hidden_layers_enc,
                hidden_layers_dec=hidden_layers_dec,
                hidden_layers_feat_dec=hidden_layers_decoding,
                feat_decoding=feature_decoding,
                target_names_and_types={"t1": output_type},
                # output_type=output_type,
                predict_all=predict_all,
                shuffle=shuffle,
                input_dims={f: [i] for i, f in enumerate(all_features)},
                output_dims={"t1": 0},
            )

        else:
            # baseline model
            h_size = 32
            hidden_layers = [
                h_size,
                h_size,
            ]

            global_model = BaselineModel(
                input_dim=len(all_features),
                feature_names=all_features,
                state_dim=state_dim,
                hidden_layers_enc=hidden_layers_enc,
                hidden_layers_dec=hidden_layers_dec ,
                hidden_layers_decoding=hidden_layers_decoding,
                feat_decoding=feature_decoding,
                # output_type=output_type,
                output_dims={"t1": 0},
                target_names_and_types={"t1": output_type},
            )
            centralized_model = BaselineModel(
                input_dim=len(all_features),
                feature_names = all_features,
                state_dim=state_dim,
                hidden_layers_enc=hidden_layers_enc,
                hidden_layers_dec=hidden_layers_dec,
                hidden_layers_decoding=hidden_layers_decoding,
                feat_decoding=feature_decoding,
                # output_type=output_type,
                output_dims={"t1": 0},
                target_names_and_types={"t1": output_type},
            )

        print(f"Number of model parameters {count_parameters(global_model)}")

        # for feature decoding, not necessarily used
        criterion_features = nn.MSELoss()

        criterion_targets = {"t1": nn.MSELoss()}

        if model_type == "modn":
            # local models trained with FL. Perform a few local training steps before aggregation of parameters with global model
            local_models = [
                ModularModel(
                    feature_names=features,
                    state_dim=state_dim,
                    hidden_layers_enc=hidden_layers_enc,
                    hidden_layers_dec=hidden_layers_dec,
                    hidden_layers_feat_dec=hidden_layers_decoding,
                    feat_decoding=feature_decoding,
                    # output_type=output_type,
                    target_names_and_types={"t1": output_type},
                    predict_all=predict_all,
                    shuffle=shuffle,
                    input_dims={f: [all_features.index(f)] for f in features},
                    output_dims={"t1": 0},
                )
                for _, _, features, _, _ in node_data_train
            ]

            # local models for (non-collaborative) training on local data (lower baseline)
            local_models_independent = [
                ModularModel(
                    feature_names=features,
                    state_dim=state_dim,
                    hidden_layers_enc=hidden_layers_enc,
                    hidden_layers_dec=hidden_layers_dec,
                    hidden_layers_feat_dec=hidden_layers_decoding,
                    feat_decoding=feature_decoding,
                    # output_type=output_type,
                    predict_all=predict_all,
                    shuffle=shuffle,
                    input_dims={f: [all_features.index(f)] for f in features},
                    output_dims={"t1": 0},
                    target_names_and_types={"t1": output_type},
                )
                for _, _, features, _, _ in node_data_train
            ]

        else:
            # baseline
            local_models = [
                BaselineModel(
                    input_dim=len(all_features),
                    feature_names = features, 
                    state_dim=state_dim,
                    hidden_layers_enc=hidden_layers_enc,
                    hidden_layers_dec=hidden_layers_dec,
                    hidden_layers_decoding=hidden_layers_decoding,
                    feat_decoding=feature_decoding,
                    # output_type=output_type,
                    output_dims={"t1": 0},
                    target_names_and_types={"t1": output_type},
                )
                for _, _, features, _, _ in node_data_train
            ]
            local_models_independent = [
                BaselineModel(
                    input_dim=len(all_features),
                    feature_names = features, 
                    state_dim=state_dim,
                    hidden_layers_enc=hidden_layers_enc,
                    hidden_layers_dec=hidden_layers_dec,
                    hidden_layers_decoding=hidden_layers_decoding,
                    feat_decoding=feature_decoding,
                    # output_type=output_type,
                    output_dims={"t1": 0},
                    target_names_and_types={"t1": output_type},
                )
                for _, _, features, _, _ in node_data_train
            ]

        for i, local_model_ind in enumerate(local_models_independent):
            local_model_ind.load_state_dict(global_model.state_dict(), strict=False)
            local_models[i].load_state_dict(global_model.state_dict(), strict=False)

        # Define loss types and corresponding datasets
        loss_types = ["fl", "local", "centralized"]
        data_dicts = [node_data_train, node_data_valid, node_data_test]

        # Initialize dict for saving losses
        # losses = {
        #     split: init_all_loss_dicts(loss_types, data_dicts)
        #     for split in ["train", "valid", "test"]
        # }

        losses = {
            split: {
                loss_type: {node_id: {"t1": []} for node_id in range(1, len(data) + 1)}
                for loss_type, data in zip(loss_types, data_dicts)
            }
            for split in ["train", "valid", "test"]
        }

        # Training federated model and local models
        for round_num in range(federated_rounds):
            print(f"--- Federated Round {round_num + 1} ---")

            # Federated learning: Train each node and aggregate
            for i_model, (X, y, feature_names, nan_mask, nan_target_mask) in enumerate(
                node_data_train
            ):
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
                # load shared parameters from global model
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
                    criterion_targets,
                    local_epochs,
                    feature_names,
                    ["t1"],
                    nan_mask,
                    nan_target_mask,
                    batch_size,
                    learning_rate,
                    feature_decoding,
                )

            # Aggregate the parameters of the local models to update the global model
            aggregate_models(
                global_model,
                local_models,
                data_config["n_samples"],
                weight=False,
                model_type=model_type,
            )

            # Local-only training: Train each independent model (lower baseline) on local data only
            for local_model_ind, (
                X,
                y,
                feature_names,
                nan_mask,
                nan_target_mask,
            ) in zip(local_models_independent, node_data_train):
                train_on_node(
                    local_model_ind,
                    global_model,
                    X,
                    y,
                    criterion_features,
                    criterion_targets,
                    local_epochs,
                    feature_names,
                    ["t1"],
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
                criterion_targets,
                local_epochs,
                all_features,
                ["t1"],
                torch.isnan(X_centralized_train),
                torch.isnan(y_centralized_train),
                batch_size,
                learning_rate,
                feature_decoding,
            )

            # Evaluate models
            global_model.eval()
            with torch.no_grad():
                for node_id, (
                    X,
                    y,
                    feature_names,
                    nan_mask,
                    nan_target_mask,
                ) in enumerate(node_data_train, 1):

                    local_model = local_models[node_id - 1]

                    local_model.eval()

                    global_dict = global_model.state_dict()
                    # Filter out private params
                    shared_global_dict = {
                        k: v for k, v in global_dict.items() if "private" not in k
                    }
                    local_dict = local_model.state_dict()
                    local_dict.update(shared_global_dict)
                    local_model.load_state_dict(local_dict, strict=False)

                    X_test, y_test, _, nan_mask_test, nan_target_mask_test = (
                        node_data_test[node_id - 1]
                    )
                    X_valid, y_valid, _, nan_mask_valid, nan_target_mask_valid = (
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
                        (
                            X_test,
                            y_test,
                            nan_mask_test,
                            nan_target_mask_test,
                        ),  # Test Data
                    ]

                    # update storing dicts
                    losses, _, _, _, _ = compute_and_store_losses(
                        local_model,
                        "fl",
                        data,
                        losses,
                        f1_scores=None,
                        auprc_scores=None,
                        auprc_scores_neg=None,
                        auc_scores=None,
                        node_id=node_id,
                        feature_names=feature_names,
                        target_names=["t1"],
                        criterion_targets=criterion_targets,
                    )
                    losses, _, _, _, _ = compute_and_store_losses(
                        local_models_independent[node_id - 1],
                        "local",
                        data,
                        losses,
                        f1_scores=None,
                        auprc_scores=None,
                        auprc_scores_neg=None,
                        auc_scores=None,
                        node_id=node_id,
                        feature_names=feature_names,
                        target_names=["t1"],
                        criterion_targets=criterion_targets,
                    )
                    losses, _, _, _, _ = compute_and_store_losses(
                        centralized_model,
                        "centralized",
                        data,
                        losses,
                        f1_scores=None,
                        auprc_scores=None,
                        auprc_scores_neg=None,
                        auc_scores=None,
                        node_id=node_id,
                        feature_names=feature_names,
                        target_names=["t1"],
                        criterion_targets=criterion_targets,
                    )

                    # Print debugging information
                    print_debugging_losses(losses, "fl", node_id)
                    print_debugging_losses(losses, "local", node_id)
                    print_debugging_losses(losses, "centralized", node_id)

        print(f"Number of model parameters {count_parameters(global_model)}")

