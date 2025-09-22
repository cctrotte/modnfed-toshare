import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# -------------------------------
# Data Generation Helpers
# -------------------------------
def generate_correlated_features(
    n_samples, n_features, random_state=None, mean=None, cov=None
):
    """
    Generate a base dataset of correlated features using a random covariance matrix.
    """
    rng = np.random.default_rng(random_state)

    if mean is None:
        # Create a random positive semi-definite covariance matrix
        A = rng.normal(size=(n_features, n_features))
        cov = np.dot(A, A.T)  # ensures symmetry and positive semi-definiteness

        # Generate random mean for the base distribution
        mean = rng.normal(loc=0.0, scale=1.0, size=n_features)

    # Sample from multivariate normal
    X = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)
    return X, mean, cov


def true_label_function_classification(X, w=None, bias=0.0, random_state=None):
    """
    Generate binary labels from X using a logistic function:
      y ~ Bernoulli(sigmoid(X @ w + bias)).
    """
    rng = np.random.default_rng(random_state)
    if w is None:
        w = rng.normal(size=X.shape[1])

    logits = X @ w + bias
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(n=1, p=prob, size=X.shape[0])
    return y, w


def true_label_function_complex(X, w_linear=None, random_state=None):
    """
    Generate binary labels from X using a more complex nonlinear function, then
    a logistic mapping:

        logits = X @ w_linear
                 + alpha * (X[:,0] * X[:,1])
                 + beta * sin(X[:,2])
                 + gamma * (X[:,3]^2)
                 + bias

    Then:
        y ~ Bernoulli(sigmoid(logits))

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    w_linear : np.ndarray or None
        Linear weight vector. If None, we sample it randomly.
    random_state : int or None
        Reproducibility seed.

    Returns
    -------
    y : np.ndarray of shape (n_samples,)
        Binary class labels in {0,1}.
    w_params : dict
        Contains the randomly generated parameters (w_linear, alpha, beta, gamma, bias).
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    # Create or sample linear weights
    if w_linear is None:
        w_linear = rng.normal(loc=0.0, scale=1.0, size=n_features)

    # Nonlinear coefficients
    alpha = 0.685  # coefficient for X[:,0]*X[:,1]
    beta = 1.234  # coefficient for sin(X[:,2])
    gamma = 0.563  # coefficient for (X[:,3]^2)
    delta = 0.472  # coefficient for cos(X[:,4])
    bias = 1.432  # bias term

    # Compute individual terms (skip if feature not present)
    logits_linear = X @ w_linear
    interaction = alpha * (X[:, 0] * X[:, 1]) if n_features > 1 else 0.0
    sin_term = beta * np.sin(X[:, 2]) if n_features > 2 else 0.0
    quad_term = gamma * (X[:, 3] ** 2) if n_features > 3 else 0.0
    cos_term = delta * np.cos(X[:, 4]) if n_features > 4 else 0.0

    # Sum everything
    logits = (
        logits_linear
        + interaction
        + sin_term
        + quad_term
        + cos_term
        + bias
        + rng.normal(loc=0, scale=1 + np.abs(X[:, 1]), size=n_samples)
    )

    # Convert to probabilities via sigmoid
    prob = 1.0 / (1.0 + np.exp(-logits))

    # Sample binary labels
    y = rng.binomial(n=1, p=prob, size=n_samples)

    w_params = {
        "w_linear": w_linear,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "bias": bias,
    }
    return y, w_params


def node_specific_shift(X, shift_type="none", random_state=None):
    """
    Apply a node-specific shift to features.
    shift_type can be 'cov', 'mean', or 'nonlinear'.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    if shift_type == "none":
        # No transformation
        X_shifted = X.copy()

    if shift_type == "cov":
        # Random linear transformation
        B = rng.normal(size=(n_features, n_features))
        X_shifted = X @ B
    elif shift_type == "mean":
        # Add a random shift in the mean
        mean_shift = rng.normal(loc=0.0, scale=3.0, size=n_features)
        X_shifted = X + mean_shift
    elif shift_type == "nonlinear":
        # Example: apply a mild nonlinear transform to half the features
        X_shifted = X.copy()
        half = n_features // 2
        X_shifted[:, :half] = np.exp(X_shifted[:, :half] / 5.0)
    else:
        # No shift
        X_shifted = X.copy()

    return X_shifted


def binarize_features(X, feature_indices, method="median"):
    """
    Binarize specific features in X based on either the median or mean threshold.
    """
    X_binarized = X.copy()
    for col in feature_indices:
        col_data = X_binarized[:, col]
        if method == "median":
            threshold = np.median(col_data)
        elif method == "mean":
            threshold = np.mean(col_data)
        else:
            raise ValueError(f"Unknown method for binarization: {method}")

        X_binarized[:, col] = (col_data >= threshold).astype(float)
    return X_binarized


def generate_federated_data_classification(
    n_nodes=3,
    n_samples=[200, 200, 200],
    n_features=5,
    shift_types=None,
    random_state=42,
    binarize_feats=None,
    binarize_method="median",
):
    """
    Generate synthetic classification data for multiple nodes, each having a different
    shift in features, but the same underlying label function (logistic).
    Optionally, binarize some subset of features after the shift is applied.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (federated clients).
    n_samples_per_node : int
        Samples per node.
    n_features : int
        Number of features for each node's data.
    shift_types : list[str]
        Types of shifts to apply (e.g. 'cov', 'mean', 'nonlinear'). Must be at least
        as long as n_nodes or will cycle.
    random_state : int
        Seed for reproducibility.
    binarize_feats : list of int or None
        Indices of features to binarize after shift (e.g., [0,2] to binarize feat 0 & 2).
    binarize_method : str
        "median" or "mean" threshold for binarization.

    Returns
    -------
    nodes_data : list of (X_node, y_node)
        Each item is a tuple of (features, labels) for that node.
    w_true : np.array
        The weight vector used for generating labels (same for all nodes).
    """
    rng = np.random.default_rng(random_state)

    if shift_types is None:
        shift_types = ["cov", "mean", "nonlinear"]

    w_true = rng.normal(loc=0.0, scale=1.0, size=n_features)  # "true" weight vector

    nodes_data = []
    mean = None
    cov = None
    for i in range(n_nodes):
        # Generate a base correlated dataset for each node
        X_base, mean, cov = generate_correlated_features(
            n_samples=n_samples[i],
            n_features=n_features,
            random_state=random_state + i,
            mean=mean,
            cov=cov,
        )

        # Apply a node-specific shift
        shift_type = shift_types[i % len(shift_types)]
        X_node = node_specific_shift(X_base, shift_type, random_state=random_state + i)

        # Optionally, binarize some features (after shift)
        if binarize_feats is not None and len(binarize_feats) > 0:
            X_node = binarize_features(X_node, binarize_feats, method=binarize_method)

        # Generate binary labels with the same w_true
        # y_node, _ = true_label_function_classification(
        #     X_node, w=w_true, bias=0.0, random_state=random_state + 1000 + i
        # )
        y_node, _ = true_label_function_complex(
            X_node, w_linear=w_true, random_state=random_state + 1000 + i
        )

        nodes_data.append((X_node, y_node))

    return nodes_data, w_true


def scale_and_plot(nodes_data, shift_types, binarize_feats):
    n_features = nodes_data[0][0].shape[1]
    # Loop through each node
    for i, (X_node, y_node) in enumerate(nodes_data):
        # -----------------------------
        # 1) Local feature scaling
        # -----------------------------
        mean_i = X_node.mean(axis=0)
        std_i = X_node.std(axis=0)
        # Avoid division by zero
        std_i[std_i < 1e-12] = 1e-12

        cols_to_scale = [i for i in range(X_node.shape[1]) if i not in binarize_feats]
        X_node_scaled = X_node.copy()

        X_node_scaled[:, cols_to_scale] = (
            X_node[:, cols_to_scale] - mean_i[cols_to_scale]
        ) / std_i[cols_to_scale]

        # Create a DataFrame for convenience in Seaborn
        col_names = [f"feat_{j}" for j in range(n_features)]
        df = pd.DataFrame(X_node_scaled, columns=col_names)
        df["label"] = y_node

        print(f"\n=== Node {i}: shift={shift_types[i]} ===")
        print("Scaled feature means:", df[col_names].mean().values.round(2))
        print("Scaled feature stds:", df[col_names].std().values.round(2))

        # -----------------------------
        # 2) Distribution plots for each feature AND the target
        # -----------------------------
        fig, axes = plt.subplots(1, n_features + 1, figsize=(4 * (n_features + 1), 4))

        # For each feature, do a histogram (or kde) color by label
        for j in range(n_features):
            ax_j = axes[j]
            sns.histplot(
                data=df,
                x=col_names[j],
                hue="label",
                element="step",
                kde=True,
                alpha=0.5,
                ax=ax_j,
            )
            ax_j.set_title(f"Node {i} - {col_names[j]}")

        # Last plot: label distribution (countplot)
        ax_label = axes[n_features]
        sns.countplot(x="label", data=df, ax=ax_label)
        ax_label.set_title(f"Node {i} - Label Distribution")

        plt.tight_layout()
        plt.savefig(f"tmp/node_{i}_dist.png")

        plt.show()

        # -----------------------------
        # 3) Pairwise scatter plots (all features)
        # -----------------------------
        pairplot_fig = sns.pairplot(df, hue="label", diag_kind="hist", corner=False)
        pairplot_fig.fig.suptitle(
            f"Node {i} - Pairwise Scatter (Scaled Features)", y=1.02
        )
        plt.savefig(f"tmp/node_{i}_pair.png")

        # -----------------------------
        # 4) (Optional) PCA scatter plot
        # -----------------------------
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(df[col_names].values)

        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], c=df["label"], cmap="bwr", alpha=0.7
        )
        handles, labels = scatter.legend_elements()
        ax.legend(handles, [f"Class {lbl}" for lbl in labels], title="Label")
        ax.set_title(f"Node {i} - PCA (Scaled Features)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        plt.savefig(f"tmp/node_{i}_pca.png")

    return
