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
from utils.data import preprocess_data_for_training_shift
from utils.data_shifts import generate_federated_data_classification, scale_and_plot
from utils.utils_epoct import get_class_weights

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

import os
from dotenv import load_dotenv


if __name__ == "__main__":

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    train_config = load_config(config_file="config_training_shift.yaml")
    data_config = load_config(config_file="config_data_shift.yaml")
    output_type = data_config["output_type"]

    config = {**train_config, **data_config}
    wandb.init(project="flmodn_synth", config=config)

    keys, values = zip(*train_config.items())
    keys_data, values_data = zip(*data_config.items())

    # all possible features sorted in alphabetical order
    all_features = sorted(
        list(
            set([f for features in data_config["node_feature_sets"] for f in features])
        )
    )
    # Parameters
    # n_samples_per_node = 100
    n_features = len(all_features)
    # shift_types = ["cov", "nonlinear", "mean", "none", "none"]
    n_nodes = len(data_config["node_feature_sets"])
    node_list = ["node_" + str(i) for i in range(1, n_nodes + 1)]

    # binarize_feats = [0, 3]

    # nonlinear and mean introduce most differences
    # Generate federated data (binary classification)
    node_data, w_true = generate_federated_data_classification(
        n_nodes=n_nodes,
        n_samples=data_config["n_samples"],
        n_features=n_features,
        shift_types=data_config["shift_types"],
        random_state=42,
        binarize_feats=data_config["binarize_feats"],
        binarize_method="median",
    )
    (
        X_centralized_train,
        X_centralized_valid,
        y_centralized_train,
        y_centralized_valid,
        node_data_train,
        node_data_valid,
        node_data_test,
    ) = preprocess_data_for_training_shift(
        node_data,
        data_config["binarize_feats"],
        all_features,
        data_config["node_feature_sets"],
        data_config["impute"],
        data_config["nan_ratio"],
        data_config["data_seed"],
    )
    scale_and_plot(node_data, data_config["shift_types"], data_config["binarize_feats"])

    target_weights = get_class_weights(
        train_config["weight"][0],
        y_centralized_train,
        node_data_train,
        ["t1"],
        node_list,
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

            # shared FL model
            global_model = ModularModel(
                feature_names=all_features,
                state_dim=state_dim,
                hidden_layers_enc=config_combi["hidden_layers_enc_shared"],
                hidden_layers_enc_private=config_combi["hidden_layers_enc_private"],
                hidden_layers_dec=config_combi["hidden_layers_dec_shared"],
                hidden_layers_dec_private=config_combi["hidden_layers_dec_private"],
                hidden_layers_feat_dec=config_combi["hidden_layers_decoding"],
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
                hidden_layers_enc=config_combi["hidden_layers_enc_shared"] + config_combi["hidden_layers_enc_private"],
                hidden_layers_dec=config_combi["hidden_layers_dec_shared"] + config_combi["hidden_layers_dec_private"],
                hidden_layers_feat_dec=config_combi["hidden_layers_decoding"],
                feat_decoding=feature_decoding,
                target_names_and_types={"t1": output_type},
                # output_type=output_type,
                predict_all=predict_all,
                shuffle=shuffle,
                input_dims={f: [i] for i, f in enumerate(all_features)},
                output_dims={"t1": 0},
            )
            centralized_model.load_state_dict(global_model.state_dict(), strict=False)

        else:
            # baseline model

            global_model = BaselineModel(
                input_dim=len(all_features),
                state_dim=state_dim,
                hidden_layers_enc=config_combi["hidden_layers_enc_private"]
                + config_combi["hidden_layers_enc_shared"],
                hidden_layers_dec=config_combi["hidden_layers_dec_private"]
                + config_combi["hidden_layers_dec_shared"],
                hidden_layers_decoding=config_combi["hidden_layers_decoding"],
                feat_decoding=feature_decoding,
                # output_type=output_type,
                output_dims={"t1": 0},
                target_names_and_types={"t1": output_type},
            )
            centralized_model = BaselineModel(
                input_dim=len(all_features),
                state_dim=state_dim,
                hidden_layers_enc=config_combi["hidden_layers_enc_private"]
                + config_combi["hidden_layers_enc_shared"],
                hidden_layers_dec=config_combi["hidden_layers_dec_private"]
                + config_combi["hidden_layers_dec_shared"],
                hidden_layers_decoding=config_combi["hidden_layers_decoding"],
                feat_decoding=feature_decoding,
                # output_type=output_type,
                output_dims={"t1": 0},
                target_names_and_types={"t1": output_type},
            )

        print(f"Number of model parameters {count_parameters(global_model)}")

        # for feature decoding, not necessarily used
        criterion_features = nn.MSELoss()

        criterion_targets = {
            node: {
                name: nn.BCEWithLogitsLoss(pos_weight=target_weights[node][name])
                for name in ["t1"]
            }
            for node in node_list
        }
        criterion_targets["centralized"] = {
            name: nn.BCEWithLogitsLoss(pos_weight=target_weights["centralized"][name])
            for name in ["t1"]
        }
        if model_type == "modn":
            # local models trained with FL. Perform a few local training steps before aggregation of parameters with global model
            local_models = [
                ModularModel(
                    feature_names=features,
                    state_dim=state_dim,
                    hidden_layers_enc=config_combi["hidden_layers_enc_shared"],
                    hidden_layers_enc_private=config_combi["hidden_layers_enc_private"],
                    hidden_layers_dec=config_combi["hidden_layers_dec_shared"],
                    hidden_layers_dec_private=config_combi["hidden_layers_dec_private"],
                    hidden_layers_feat_dec=config_combi["hidden_layers_decoding"],
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
                    hidden_layers_enc=config_combi["hidden_layers_enc_shared"] + config_combi["hidden_layers_enc_private"],
                    hidden_layers_dec=config_combi["hidden_layers_dec_shared"] + config_combi["hidden_layers_dec_private"],
                    hidden_layers_feat_dec=config_combi["hidden_layers_decoding"],
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
                    state_dim=state_dim,
                    hidden_layers_enc=config_combi["hidden_layers_enc_private"]
                    + config_combi["hidden_layers_enc_shared"],
                    hidden_layers_dec=config_combi["hidden_layers_dec_private"]
                    + config_combi["hidden_layers_dec_shared"],
                    hidden_layers_decoding=config_combi["hidden_layers_decoding"],
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
                    state_dim=state_dim,
                    hidden_layers_enc=config_combi["hidden_layers_enc_private"]
                    + config_combi["hidden_layers_enc_shared"],
                    hidden_layers_dec=config_combi["hidden_layers_dec_private"]
                    + config_combi["hidden_layers_dec_shared"],
                    hidden_layers_decoding=config_combi["hidden_layers_decoding"],
                    feat_decoding=feature_decoding,
                    # output_type=output_type,
                    output_dims={"t1": 0},
                    target_names_and_types={"t1": output_type},
                )
                for _, _, features, _, _ in node_data_train
            ]

        for i, local_model in enumerate(local_models):
            local_model.load_state_dict(global_model.state_dict(), strict=False)

        # Define loss types and corresponding datasets
        loss_types = ["fl", "local", "centralized"]
        data_dicts = [node_data_train, node_data_valid, node_data_test]

        # Initialize dict for saving losses

        losses = {
            split: {
                loss_type: {node_id: {"t1": []} for node_id in range(1, len(data) + 1)}
                for loss_type, data in zip(loss_types, data_dicts)
            }
            for split in ["train", "valid", "test"]
        }

        f1_scores = {
            split: {
                loss_type: {node_id: {"t1": []} for node_id in range(1, len(data) + 1)}
                for loss_type, data in zip(loss_types, data_dicts)
            }
            for split in ["train", "valid", "test"]
        }

        auprc_scores = {
            split: {
                loss_type: {node_id: {"t1": []} for node_id in range(1, len(data) + 1)}
                for loss_type, data in zip(loss_types, data_dicts)
            }
            for split in ["train", "valid", "test"]
        }
        auprc_scores_neg = {
            split: {
                loss_type: {node_id: {"t1": []} for node_id in range(1, len(data) + 1)}
                for loss_type, data in zip(loss_types, data_dicts)
            }
            for split in ["train", "valid", "test"]
        }
        auc_scores = {
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
            for i_model, (
                node,
                (X, y, feature_names, nan_mask, nan_target_mask),
            ) in enumerate(zip(node_list, node_data_train)):
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
                    X,
                    y,
                    criterion_features,
                    criterion_targets[node],
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
                weight=True,
                model_type=model_type,
            )

            # global_dict = global_model.state_dict()
            # # Filter out private params
            # shared_global_dict = {
            #     k: v for k, v in global_dict.items() if "private" not in k
            # }
            # for local_model in local_models:
            #     local_dict = local_model.state_dict()
            #     local_dict.update(shared_global_dict)
            #     local_model.load_state_dict(local_dict, strict=False)

            # Local-only training: Train each independent model (lower baseline) on local data only
            for (
                local_model_ind,
                node,
                (
                    X,
                    y,
                    feature_names,
                    nan_mask,
                    nan_target_mask,
                ),
            ) in zip(local_models_independent, node_list, node_data_train):
                train_on_node(
                    local_model_ind,
                    X,
                    y,
                    criterion_features,
                    criterion_targets[node],
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
                X_centralized_train,
                y_centralized_train,
                criterion_features,
                criterion_targets["centralized"],
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
                    node,
                    (
                        X,
                        y,
                        feature_names,
                        nan_mask,
                        nan_target_mask,
                    ),
                ) in enumerate(zip(node_list, node_data_train), 1):

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
                    losses, f1_scores, auprc_scores, auprc_scores_neg, auc_scores = (
                        compute_and_store_losses(
                            local_model,
                            "fl",
                            data,
                            losses,
                            f1_scores=f1_scores,
                            auprc_scores=auprc_scores,
                            auprc_scores_neg=auprc_scores_neg,
                            auc_scores=auc_scores,
                            node_id=node_id,
                            feature_names=feature_names,
                            target_names=["t1"],
                            criterion_targets=criterion_targets[node],
                        )
                    )
                    losses, f1_scores, auprc_scores, auprc_scores_neg, auc_scores = (
                        compute_and_store_losses(
                            local_models_independent[node_id - 1],
                            "local",
                            data,
                            losses,
                            f1_scores=f1_scores,
                            auprc_scores=auprc_scores,
                            auprc_scores_neg=auprc_scores_neg,
                            auc_scores=auc_scores,
                            node_id=node_id,
                            feature_names=feature_names,
                            target_names=["t1"],
                            criterion_targets=criterion_targets[node],
                        )
                    )
                    losses, f1_scores, auprc_scores, auprc_scores_neg, auc_scores = (
                        compute_and_store_losses(
                            centralized_model,
                            "centralized",
                            data,
                            losses,
                            f1_scores=f1_scores,
                            auprc_scores=auprc_scores,
                            auprc_scores_neg=auprc_scores_neg,
                            auc_scores=auc_scores,
                            node_id=node_id,
                            feature_names=feature_names,
                            target_names=["t1"],
                            criterion_targets=criterion_targets[node],
                        )
                    )

                    # Print debugging information
                    print_debugging_losses(losses, "fl", node_id)
                    print_debugging_losses(losses, "local", node_id)
                    print_debugging_losses(losses, "centralized", node_id)

                wandb_valid_losses = {
                    f"average valid loss {t} fl": np.mean(
                        [
                            losses["valid"]["fl"][node_id][t][-1]
                            for node_id in range(1, len(node_list) + 1)
                        ]
                    )
                    for t in ["t1"]
                }
                wandb_valid_losses.update(
                    {
                        f"average valid loss {t} local": np.mean(
                            [
                                losses["valid"]["local"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                wandb_valid_losses.update(
                    {
                        f"average valid loss {t} centralized": np.mean(
                            [
                                losses["valid"]["centralized"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                # wandb_valid_losses.update({f"average valid loss all fl": np.average([np.mean([losses["valid"]["fl"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_losses.update({f"average valid loss all local": np.average([np.mean([losses["valid"]["local"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_losses.update({f"average valid loss all centralized": np.average([np.mean([losses["valid"]["centralized"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                wandb_valid_losses.update({"fl round": round_num})
                wandb.log(wandb_valid_losses)

                wandb_valid_f1 = {
                    f"average valid f1 {t} fl": np.mean(
                        [
                            f1_scores["valid"]["fl"][node_id][t][-1]
                            for node_id in range(1, len(node_list) + 1)
                        ]
                    )
                    for t in ["t1"]
                }
                wandb_valid_f1.update(
                    {
                        f"average valid f1 {t} local": np.mean(
                            [
                                f1_scores["valid"]["local"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                wandb_valid_f1.update(
                    {
                        f"average valid f1 {t} centralized": np.mean(
                            [
                                f1_scores["valid"]["centralized"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                # wandb_valid_f1.update({f"average valid f1 all fl": np.average([np.mean([f1_scores["valid"]["fl"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_f1.update({f"average valid f1 all local": np.average([np.mean([f1_scores["valid"]["local"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_f1.update({f"average valid f1 all centralized": np.average([np.mean([f1_scores["valid"]["centralized"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                wandb_valid_f1.update({"fl round": round_num})
                wandb.log(wandb_valid_f1)

                wandb_valid_au = {
                    f"average valid au {t} fl": np.mean(
                        [
                            auprc_scores["valid"]["fl"][node_id][t][-1]
                            for node_id in range(1, len(node_list) + 1)
                        ]
                    )
                    for t in ["t1"]
                }
                wandb_valid_au.update(
                    {
                        f"average valid au {t} local": np.mean(
                            [
                                auprc_scores["valid"]["local"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                wandb_valid_au.update(
                    {
                        f"average valid au {t} centralized": np.mean(
                            [
                                auprc_scores["valid"]["centralized"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                # wandb_valid_au.update({f"average valid au all fl": np.average([np.mean([auprc_scores["valid"]["fl"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_au.update({f"average valid au all local": np.average([np.mean([auprc_scores["valid"]["local"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_au.update({f"average valid au all centralized": np.average([np.mean([auprc_scores["valid"]["centralized"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                wandb_valid_au.update({"fl round": round_num})
                wandb.log(wandb_valid_au)

                wandb_valid_au_neg = {
                    f"average valid au neg {t} fl": np.mean(
                        [
                            auprc_scores_neg["valid"]["fl"][node_id][t][-1]
                            for node_id in range(1, len(node_list) + 1)
                        ]
                    )
                    for t in ["t1"]
                }
                wandb_valid_au_neg.update(
                    {
                        f"average valid au neg {t} local": np.mean(
                            [
                                auprc_scores_neg["valid"]["local"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                wandb_valid_au_neg.update(
                    {
                        f"average valid au neg {t} centralized": np.mean(
                            [
                                auprc_scores_neg["valid"]["centralized"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                # wandb_valid_au_neg.update({f"average valid au neg all fl": np.average([np.mean([auprc_scores_neg["valid"]["fl"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_au_neg.update({f"average valid au neg all local": np.average([np.mean([auprc_scores_neg["valid"]["local"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_valid_au_neg.update({f"average valid au neg all centralized": np.average([np.mean([auprc_scores_neg["valid"]["centralized"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                wandb_valid_au_neg.update({"fl round": round_num})
                wandb.log(wandb_valid_au_neg)

                wandb_train_losses = {
                    f"average train loss {t} fl": np.mean(
                        [
                            losses["train"]["fl"][node_id][t][-1]
                            for node_id in range(1, len(node_list) + 1)
                        ]
                    )
                    for t in ["t1"]
                }
                wandb_train_losses.update(
                    {
                        f"average train loss {t} local": np.mean(
                            [
                                losses["train"]["local"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                wandb_train_losses.update(
                    {
                        f"average train loss {t} centralized": np.mean(
                            [
                                losses["train"]["centralized"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                # wandb_train_losses.update({f"average train loss all fl": np.average([np.mean([losses["train"]["fl"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_train_losses.update({f"average train loss all local": np.average([np.mean([losses["train"]["local"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_train_losses.update({f"average train loss all centralized": np.average([np.mean([losses["train"]["centralized"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                wandb_train_losses.update({"fl round": round_num})
                wandb.log(wandb_train_losses)

                wandb_train_f1 = {
                    f"average train f1 {t} fl": np.mean(
                        [
                            f1_scores["train"]["fl"][node_id][t][-1]
                            for node_id in range(1, len(node_list) + 1)
                        ]
                    )
                    for t in ["t1"]
                }
                wandb_train_f1.update(
                    {
                        f"average train f1 {t} local": np.mean(
                            [
                                f1_scores["train"]["local"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                wandb_train_f1.update(
                    {
                        f"average train f1 {t} centralized": np.mean(
                            [
                                f1_scores["train"]["centralized"][node_id][t][-1]
                                for node_id in range(1, len(node_list) + 1)
                            ]
                        )
                        for t in ["t1"]
                    }
                )
                # wandb_train_f1.update({f"average train f1 all fl": np.average([np.mean([f1_scores["train"]["fl"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_train_f1.update({f"average train f1 all local": np.average([np.mean([f1_scores["train"]["local"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                # wandb_train_f1.update({f"average train f1 all centralized": np.average([np.mean([f1_scores["train"]["centralized"][node_id][t][-1] for t in ["t1"]]) for node_id in range(1, len(node_list) + 1)])})
                wandb_train_f1.update({"fl round": round_num})
                wandb.log(wandb_train_f1)

        # some plots and save results
        bsl_prediction = plot_losses_per_node(
            losses, node_data_train, node_data_test, model_type
        )

        plot_f1_scores_per_node(f1_scores, node_data_train, model_type)

        plot_f1_scores_per_node_and_target(
            ["t1"], f1_scores, node_data_train, model_type
        )
        plot_losses_per_node_and_target(
            ["t1"], losses, node_data_train, node_data_test, model_type
        )

        print(f"Number of model parameters {count_parameters(global_model)}")
        avg_losses, std_losses = compute_avg_std(losses)
        plot_avg_losses(avg_losses, std_losses, federated_rounds, model_type)
