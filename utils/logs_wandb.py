import wandb
import numpy as np        


def log_wandb_epoct(losses, node_variables, all_targets, f1_scores, auprc_scores, auprc_scores_neg, auc_scores, round_num):
    wandb_valid_losses = {f"average valid loss {t} fl": np.mean([losses["valid"]["fl"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets}
    wandb_valid_losses.update({f"average valid loss {t} local": np.mean([losses["valid"]["local"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_losses.update({f"average valid loss {t} centralized": np.mean([losses["valid"]["centralized"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_losses.update({f"average valid loss all fl": np.average([np.mean([losses["valid"]["fl"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_losses.update({f"average valid loss all local": np.average([np.mean([losses["valid"]["local"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_losses.update({f"average valid loss all centralized": np.average([np.mean([losses["valid"]["centralized"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    
    wandb_valid_losses.update({"fl round": round_num})
    wandb.log(wandb_valid_losses)

    wandb_valid_f1 = {f"average valid f1 {t} fl": np.mean([f1_scores["valid"]["fl"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets}
    wandb_valid_f1.update({f"average valid f1 {t} local": np.mean([f1_scores["valid"]["local"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_f1.update({f"average valid f1 {t} centralized": np.mean([f1_scores["valid"]["centralized"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_f1.update({f"average valid f1 all fl": np.average([np.mean([f1_scores["valid"]["fl"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_f1.update({f"average valid f1 all local": np.average([np.mean([f1_scores["valid"]["local"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_f1.update({f"average valid f1 all centralized": np.average([np.mean([f1_scores["valid"]["centralized"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})

    wandb_valid_f1.update({"fl round": round_num})
    wandb.log(wandb_valid_f1)

    wandb_valid_au = {f"average valid au {t} fl": np.mean([auprc_scores["valid"]["fl"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets}
    wandb_valid_au.update({f"average valid au {t} local": np.mean([auprc_scores["valid"]["local"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_au.update({f"average valid au {t} centralized": np.mean([auprc_scores["valid"]["centralized"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_au.update({f"average valid au all fl": np.average([np.mean([auprc_scores["valid"]["fl"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_au.update({f"average valid au all local": np.average([np.mean([auprc_scores["valid"]["local"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_au.update({f"average valid au all centralized": np.average([np.mean([auprc_scores["valid"]["centralized"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})

    wandb_valid_au.update({"fl round": round_num})
    wandb.log(wandb_valid_au)

    wandb_valid_au_neg = {f"average valid au neg {t} fl": np.mean([auprc_scores_neg["valid"]["fl"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets}
    wandb_valid_au_neg.update({f"average valid au neg {t} local": np.mean([auprc_scores_neg["valid"]["local"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_au_neg.update({f"average valid au neg {t} centralized": np.mean([auprc_scores_neg["valid"]["centralized"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_au_neg.update({f"average valid au neg all fl": np.average([np.mean([auprc_scores_neg["valid"]["fl"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_au_neg.update({f"average valid au neg all local": np.average([np.mean([auprc_scores_neg["valid"]["local"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_au_neg.update({f"average valid au neg all centralized": np.average([np.mean([auprc_scores_neg["valid"]["centralized"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})

    wandb_valid_au_neg.update({"fl round": round_num})
    wandb.log(wandb_valid_au_neg)

    wandb_valid_auc = {f"average valid auc {t} fl": np.mean([auc_scores["valid"]["fl"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets}
    wandb_valid_auc.update({f"average valid auc {t} local": np.mean([auc_scores["valid"]["local"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_auc.update({f"average valid auc {t} centralized": np.mean([auc_scores["valid"]["centralized"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_valid_auc.update({f"average valid auc all fl": np.average([np.mean([auc_scores["valid"]["fl"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_auc.update({f"average valid auc all local": np.average([np.mean([auc_scores["valid"]["local"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_valid_auc.update({f"average valid auc all centralized": np.average([np.mean([auc_scores["valid"]["centralized"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})

    wandb_valid_auc.update({"fl round": round_num})
    wandb.log(wandb_valid_auc)

    wandb_train_losses = {f"average train loss {t} fl": np.mean([losses["train"]["fl"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets}
    wandb_train_losses.update({f"average train loss {t} local": np.mean([losses["train"]["local"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_train_losses.update({f"average train loss {t} centralized": np.mean([losses["train"]["centralized"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_train_losses.update({f"average train loss all fl": np.average([np.mean([losses["train"]["fl"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_train_losses.update({f"average train loss all local": np.average([np.mean([losses["train"]["local"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_train_losses.update({f"average train loss all centralized": np.average([np.mean([losses["train"]["centralized"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})

    wandb_train_losses.update({"fl round": round_num})
    wandb.log(wandb_train_losses)

    wandb_train_f1 = {f"average train f1 {t} fl": np.mean([f1_scores["train"]["fl"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets}
    wandb_train_f1.update({f"average train f1 {t} local": np.mean([f1_scores["train"]["local"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_train_f1.update({f"average train f1 {t} centralized": np.mean([f1_scores["train"]["centralized"][node_id][t][-1] for node_id in range(1, len(node_variables) + 1) if t in losses["valid"]["fl"][node_id].keys()]) for t in all_targets})
    wandb_train_f1.update({f"average train f1 all fl": np.average([np.mean([f1_scores["train"]["fl"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_train_f1.update({f"average train f1 all local": np.average([np.mean([f1_scores["train"]["local"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})
    wandb_train_f1.update({f"average train f1 all centralized": np.average([np.mean([f1_scores["train"]["centralized"][node_id][t][-1] for t in all_targets if t in losses["valid"]["fl"][node_id].keys()]) for node_id in range(1, len(node_variables) + 1)])})

    wandb_train_f1.update({"fl round": round_num})
    wandb.log(wandb_train_f1)

    
