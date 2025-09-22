import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yaml
import os
import wandb
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
import plotly.graph_objects as go
import plotly


def save_results_to_csv(config, results, filename="experiment_log.csv"):
    # Combine config and results into a single dictionary
    data = {**config, **results}
    df = pd.DataFrame([data])

    if os.path.exists(filename):

        existing_df = pd.read_csv(filename)
        # Identify missing columns in the existing CSV
        missing_columns = [col for col in df.columns if col not in existing_df.columns]

        # Add missing columns to the existing DataFrame with NaN values
        for col in missing_columns:
            existing_df[col] = None

        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="w", header=True, index=False)


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def print_debugging_losses(losses, training_type, node_id):
    """Print debugging target losses"""
    print(f"\nNode {node_id} - {training_type.capitalize()} Model")

    # Extract stored losses
    losses_train = losses["train"][training_type][node_id]
    losses_test = losses["test"][training_type][node_id]

    if not losses_train or not losses_test:
        print("No loss data available.")
        return

    print(
        f"{training_type.capitalize()} Debugging Losses: Step {len(losses_train)} - "
        f"Mean Target Loss (train): {np.mean([elem[-1] for elem in losses_train.values() if elem]):.4f}, Mean Target Loss (test): {np.mean([elem[-1] for elem in losses_test.values() if elem]):.4f}"
    )


def compute_and_store_losses(
    model,
    training_type,
    data,
    losses,
    f1_scores,
    auprc_scores,
    auprc_scores_neg,
    auc_scores,
    node_id,
    feature_names,
    target_names,
    criterion_targets,
    plot_auprc=False,
):
    """Computes and stores losses"""
    loss_types = ["train", "valid", "test"]

    for loss_type, (X, y, nan_mask, nan_mask_target) in zip(loss_types, data):
        l, targets = model.compute_final_loss(
            X,
            y,
            target_names,
            nan_mask_target,
            criterion_targets,
        )

        for target in l.keys():

            losses[loss_type][training_type][node_id][target].append(l[target])
            # wandb.log({f"{node_id}_{training_type}_{loss_type}_loss_{target}": l[target]})
            if f1_scores:
                y_pred = targets[target]["prediction"]
                y_true = targets[target]["ground_truth"]
                y_scores = targets[target]["scores"]
                f = f1_score(y_true, y_pred, average="macro")
                a = average_precision_score(y_true, y_scores)
                a_neg = average_precision_score(y_true, y_scores, pos_label=0)
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_scores)
                else:
                    auc = np.nan

                f1_scores[loss_type][training_type][node_id][target].append(f)
                # wandb.log({f"{node_id}_{training_type}_{loss_type}_f1_{target}": f[target]})
                auprc_scores[loss_type][training_type][node_id][target].append(a)
                auprc_scores_neg[loss_type][training_type][node_id][target].append(
                    a_neg
                )
                auc_scores[loss_type][training_type][node_id][target].append(auc)
                if plot_auprc:
                    # plot precision-recall curve
                    precision, recall, thresholds = precision_recall_curve(
                        y_true, y_scores
                    )
                    # fig, ax = plt.subplots()

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=recall,
                            y=precision,
                            mode="lines",
                            name=f"PR Curve (AUPRC = {a:.4f})",
                            line=dict(width=3),
                        )
                    )

                    fig.update_layout(
                        title="Precision-Recall Curve",
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                        template="plotly_white",
                        showlegend=True,
                    )
                    # fig.to_dict()
                    wandb.log(
                        {
                            f"{node_id}_{training_type}_{loss_type}_pr_{target}": wandb.Plotly(
                                fig
                            )
                        }
                    )
                #     fig.write_html(f"tmp/{node_id}_{training_type}_{loss_type}_pr_{target}.html")

                # plt.plot(recall, precision)

                # wandb.log({f"{node_id}_{training_type}_{loss_type}_pr_{target}": wandb.Image(fig)})
                # if plot_auprc:
                #     precision = a[1][target]
                #     recall = a[2][target]
                #     thresholds = a[3][target]
                #     # plot precision-recall curve
                #     wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})

                #     plt.figure()
                #     plt.plot(recall, precision, marker='.')
                #     plt.xlabel('Recall')
                #     plt.ylabel('Precision')
                #     plt.title(f'Precision-Recall curve for {target}')

    return losses, f1_scores, auprc_scores, auprc_scores_neg, auc_scores


def plot_losses_per_node(
    losses,
    node_data_train,
    node_data_test,
    model_type,
):
    num_nodes = len(node_data_train)
    fig, axes = plt.subplots(1, num_nodes, figsize=(24, 8))
    if num_nodes == 1:
        axes = [axes]

    for node_id in range(1, num_nodes + 1):
        ax = axes[node_id - 1]

        # Plot losses for all models
        for training_type, linestyle in [
            ("fl", "-"),
            ("local", "--"),
            ("centralized", ":"),
        ]:
            for split in ["train", "valid", "test"]:
                ax.plot(
                    np.mean(
                        [
                            elem
                            for elem in losses[split][training_type][node_id].values()
                            if elem
                        ],
                        axis=0,
                    ),
                    label=f"{training_type.capitalize()} {split.capitalize()} Loss",
                    linestyle=linestyle,
                )

        # Baseline MSE computation
        y_train_mean = torch.mean(node_data_train[node_id - 1][1])
        mask_ = ~node_data_test[node_id - 1][-3]
        targets = node_data_test[node_id - 1][1][mask_]
        bsl_prediction = nn.BCEWithLogitsLoss()(
            torch.rand(targets.shape),
            targets.float(),
        )

        ax.axhline(
            y=bsl_prediction.item(), color="r", linestyle="-", label="Naive Baseline"
        )
        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Loss")
        ax.set_title(f"Node {node_id + 1} Losses")
        ax.set_ylim((0, bsl_prediction.item() + 1))
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"federated_vs_local_training_losses_{model_type}_tmp.png")
    plt.show()
    return bsl_prediction


def plot_losses_per_node_and_target(
    all_targets, losses, node_data_train, node_data_test, model_type
):
    num_nodes = len(node_data_train)

    num_targets = len(all_targets)
    fig, axes = plt.subplots(
        num_nodes, num_targets, figsize=(5 * num_targets, 4 * num_nodes), squeeze=False
    )

    for node_idx in range(num_nodes):
        node_id = node_idx + 1

        test_targets = node_data_test[node_idx][1]
        test_mask = ~node_data_test[node_idx][4]  # invert mask if needed

        for target_idx, target in enumerate(all_targets):
            ax = axes[node_idx][target_idx]

            # Plot loss curves for every training type and split.
            for training_type, linestyle in [
                ("fl", "-"),
                ("local", "--"),
                ("centralized", ":"),
            ]:
                for split in ["train", "valid", "test"]:
                    loss_curve = losses[split][training_type][node_id].get(target)
                    if loss_curve is not None and len(loss_curve) > 0:
                        ax.plot(
                            loss_curve,
                            label=f"{training_type.capitalize()} {split.capitalize()}",
                            linestyle=linestyle,
                        )

            target_values = test_targets[test_mask[:, target_idx], target_idx]
            if len(target_values) > 0:
                # Create a baseline prediction (using random predictions) and compute the BCEWithLogitsLoss.
                bsl_loss = nn.BCEWithLogitsLoss()(
                    torch.rand(target_values.shape), target_values.float()
                )
                ax.axhline(
                    y=bsl_loss.item(), color="r", linestyle="-", label="Naive Baseline"
                )
                ax.set_ylim((0, bsl_loss.item() + 1))

            ax.set_xlabel("Federated Round")
            ax.set_ylabel("Loss")
            ax.set_title(f"Node {node_id} - Target {target}")
            ax.legend(fontsize="small", loc="best")

    plt.tight_layout()
    plt.savefig(f"losses_{model_type}_tmp.png")
    plt.show()

    return


def plot_f1_scores_per_node(f1_scores, node_data_train, model_type):
    num_nodes = len(node_data_train)
    fig, axes = plt.subplots(1, num_nodes, figsize=(24, 8))

    # Ensure axes is always iterable (i.e., a list) even if there is only one node.
    if num_nodes == 1:
        axes = [axes]

    for node_idx in range(num_nodes):
        node_id = node_idx + 1  # Assuming node IDs start at 1
        ax = axes[node_idx]

        # Plot F1 scores for each training type and split using different linestyles
        for training_type, linestyle in [
            ("fl", "-"),
            ("local", "--"),
            ("centralized", ":"),
        ]:
            for split in ["train", "valid", "test"]:
                ax.plot(
                    np.mean(
                        [
                            elem
                            for elem in f1_scores[split][training_type][
                                node_id
                            ].values()
                            if elem
                        ],
                        axis=0,
                    ),
                    label=f"{training_type.capitalize()} {split.capitalize()} F1 Score",
                    linestyle=linestyle,
                )

        ax.set_xlabel("Federated Round")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Node {node_id} F1 Scores")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"federated_vs_local_training_f1_scores_{model_type}_tmp.png")
    plt.show()


def plot_f1_scores_per_node_and_target(
    all_targets, f1_scores, node_data_train, model_type
):
    num_nodes = len(node_data_train)

    num_targets = len(all_targets)

    # Create a grid: one row per node, one column per target.
    fig, axes = plt.subplots(
        num_nodes, num_targets, figsize=(5 * num_targets, 4 * num_nodes), squeeze=False
    )

    # Iterate over nodes and targets
    for node_idx in range(num_nodes):
        node_id = node_idx + 1
        for target_idx, target in enumerate(all_targets):
            ax = axes[node_idx][target_idx]

            # Loop through each training type and split.
            for training_type, linestyle in [
                ("fl", "-"),
                ("local", "--"),
                ("centralized", ":"),
            ]:
                for split in ["train", "valid", "test"]:
                    # Retrieve the F1 scores for the specific node, training type, split, and target.
                    # It is assumed that f1_scores[split][training_type][node_id] is a dict with target keys.
                    scores = f1_scores[split][training_type][node_id].get(target)
                    if scores is not None and len(scores) > 0:
                        ax.plot(
                            scores,
                            label=f"{training_type.capitalize()} {split.capitalize()}",
                            linestyle=linestyle,
                        )

            ax.set_xlabel("Federated Round")
            ax.set_ylabel("F1 Score")
            ax.set_title(f"Node {node_id} - Target {target}")
            ax.legend(fontsize="small", loc="best")

    plt.tight_layout()
    plt.savefig(f"f1_scores_{model_type}_tmp.png")
    plt.show()


def init_all_loss_dicts(loss_types, data_dicts):

    # {
    #     loss_type: {node_id: [] for node_id in range(1, len(data) + 1)}
    #     for loss_type, data in zip(loss_types, data_dicts)
    # }

    return {
        loss_type: {
            node_id: {target: [] for target in data[node_id - 1][3]}
            for node_id in range(1, len(data) + 1)
        }
        for loss_type, data in zip(loss_types, data_dicts)
    }


def compute_avg_std(losses):
    avg, std = {}, {}
    for split in losses:
        avg[split], std[split] = {}, {}
        for model_type in losses[split]:
            avg[split][model_type] = np.mean(
                [
                    np.mean(
                        [
                            elem
                            for elem in losses[split][model_type][node].values()
                            if elem
                        ],
                        axis=0,
                    )
                    for node in losses[split][model_type].keys()
                ],
                axis=0,
            )
            std[split][model_type] = np.std(
                [
                    np.mean(
                        [
                            elem
                            for elem in losses[split][model_type][node].values()
                            if elem
                        ],
                        axis=0,
                    )
                    for node in losses[split][model_type].keys()
                ],
                axis=0,
            )
    return avg, std


def plot_avg_losses(avg_losses, std_losses, federated_rounds, model_type):
    plt.figure(figsize=(12, 8))

    linestyles = {"fl": "-", "local": "--", "centralized": ":"}

    for training_type, linestyle in linestyles.items():
        for split in ["train", "valid", "test"]:
            label = f"Avg {training_type.capitalize()} {split.capitalize()} Loss"
            plt.errorbar(
                range(federated_rounds),
                avg_losses[split][training_type],
                yerr=std_losses[split][training_type],
                label=label,
                linestyle=linestyle,
                capsize=3,
            )

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.ylim((-0.1, 1.5))
    plt.title("Average Train, Test, and Validation Loss Across Nodes per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"average_losses_across_nodes_{model_type}_pers.png")
    plt.show()
