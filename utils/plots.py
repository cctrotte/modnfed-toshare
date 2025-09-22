import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_heatmap(pred, gt, all_targets, feats_to_keep, node_features, feat_order, patient_feat_unsc, model, save_path = "heatmap.png"):
    plt.figure(figsize=(20, 10))
    gt = ["No" if t == 0 else "Yes" if t==1 else np.nan for t in gt]
    gt_labels = [f"{t} ({gt[j]})" for j,t in enumerate(all_targets) if t in model.target_names]
    to_keep = [True] + [True  if node_features[j] in feats_to_keep else False for j in np.array(feat_order)]
    x_labels = ["No feature"] + [f"{node_features[j]}: {patient_feat_unsc[node_features[j]]}" for j in np.array(feat_order) if node_features[j] in feats_to_keep]
    ax = sns.heatmap(pred[:,to_keep], annot=True, fmt=".2f",     cmap="RdBu_r",  vmin=0,  vmax=1,  yticklabels=gt_labels, xticklabels=x_labels, cbar_kws={"label": "Predicted Probability"}, linewidths=0.5, linecolor='black', cbar=True)
    #plt.yticks(rotation=180)  # Rotate the y-axis tick labels by 90 degrees

    plt.title("Evolving Predictions as Features are Encoded")
    plt.ylabel("Predicted Target")
    plt.xlabel("Feature Encoding Order")
    plt.tight_layout()  # Adjust layout to fit everything nicely

    plt.savefig(save_path, dpi=300)
    return