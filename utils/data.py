import torch
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_data_for_nodes(
    all_features, node_feature_sets, n_samples, impute, output_type, nan_ratio, seed
):
    "generate synthetic data for each node"
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mean = torch.zeros(len(all_features))  # Mean vector
    cov = torch.eye(len(all_features))  # Identity covariance matrix
    X_test_ = torch.distributions.MultivariateNormal(mean, cov).sample((100,))
    node_data_train = []
    node_data_test = []
    node_data_valid = []
    node_data_train_imp = []
    node_data_test_imp = []
    node_data_valid_imp = []

    for i, features in enumerate(node_feature_sets):
        dep = "complex"
        X_full = torch.distributions.MultivariateNormal(mean, cov).sample(
            (n_samples[i],)
        )
        X_full_valid = torch.distributions.MultivariateNormal(mean, cov).sample(
            (n_samples[i],)
        )
        y, X_full = generate_y(
            X_full,
            dependency_type=dep,
            output_type=output_type,
        )
        y_valid, X_full_valid = generate_y(
            X_full_valid,
            dependency_type=dep,
            output_type=output_type,
        )
        y_test, X_full_test = generate_y(
            X_test_.clone(),
            dependency_type=dep,
            output_type=output_type,
        )

        if output_type == "continuous":
            y_mean = y.mean()
            y_std = y.std()
            y = (y - y_mean) / y_std
            y_valid = (y_valid - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

        # select available features for each node
        X_node = select_features(
            X_full,
            features,
            all_features,
        )
        X_node_valid = select_features(
            X_full_valid,
            features,
            all_features,
        )
        X_node_test = select_features(
            X_full_test,
            features,
            all_features,
        )

        avail_features = features if not impute else all_features
        # introduce some missingness
        nan_ratio_ = nan_ratio[i]
        X_node = introduce_nans(X_node, nan_ratio=nan_ratio_)
        X_node_valid = introduce_nans(X_node_valid, nan_ratio=nan_ratio_)
        X_node_test = introduce_nans(X_node_test, nan_ratio=nan_ratio_)

        node_data_train.append(
            (X_node, y, avail_features, torch.isnan(X_node), torch.isnan(y))
        )
        node_data_valid.append(
            (
                X_node_valid,
                y_valid,
                avail_features,
                torch.isnan(X_node_valid),
                torch.isnan(y_valid),
            )
        )
        node_data_test.append(
            (
                X_node_test,
                y_test,
                avail_features,
                torch.isnan(X_node_test),
                torch.isnan(y_test),
            )
        )

    X_centralized_train = torch.cat(
        [data[0].clone().detach() for data in node_data_train]
    )
    X_centralized_valid = torch.cat(
        [data[0].clone().detach() for data in node_data_valid]
    )
    # impute missing data for baseline
    if impute:
        for i, (X_train, y_train, features, _, _) in enumerate(node_data_train):
            (
                X_train_imp,
                X_valid_imp,
                X_test_imp,
                nan_mask,
                nan_mask_valid,
                nan_mask_test,
            ) = impute_nans(X_train, node_data_valid[i][0], node_data_test[i][0])
            node_data_train_imp.append(
                (X_train_imp, y_train, features, nan_mask, torch.isnan(y_train))
            )
            node_data_test_imp.append(
                (
                    X_test_imp,
                    node_data_test[i][1],
                    features,
                    nan_mask_test,
                    torch.isnan(node_data_test[i][1]),
                )
            )
            node_data_valid_imp.append(
                (
                    X_valid_imp,
                    node_data_valid[i][1],
                    features,
                    nan_mask_valid,
                    torch.isnan(node_data_valid[i][1]),
                )
            )
        X_centralized_train, X_centralized_valid, _, _, _, _ = impute_nans(
            X_centralized_train, X_centralized_valid
        )
    else:
        node_data_train_imp = node_data_train
        node_data_valid_imp = node_data_valid
        node_data_test_imp = node_data_test

    y_centralized_train = torch.cat([data[1] for data in node_data_train_imp], dim=0)
    y_centralized_valid = torch.cat([data[1] for data in node_data_valid_imp], dim=0)

    return (
        X_centralized_train,
        X_centralized_valid,
        y_centralized_train,
        y_centralized_valid,
        node_data_train_imp,
        node_data_valid_imp,
        node_data_test_imp,
    )


def introduce_nans(data, nan_ratio=0.1):
    """
    Randomly sets a specified ratio of values to NaN in the given data tensor or DataFrame.

    Parameters:
    - data (torch.Tensor or pandas.DataFrame): The input data where NaNs will be introduced.
    - nan_ratio (float or list of floats): The ratio of values to set to NaN. If a single float is provided,
      the same ratio is applied to all columns. If a list is provided, each column will have NaNs introduced
      according to the corresponding ratio in the list.

    Returns:
    - torch.Tensor or pandas.DataFrame: The modified data with NaNs introduced.
    """
    if isinstance(data, torch.Tensor):
        is_tensor = True
        data_np = data.numpy().copy()
    elif isinstance(data, pd.DataFrame):
        is_tensor = False
        data_np = data.values.copy()
    else:
        raise TypeError(
            "Input data must be either a torch.Tensor or a pandas.DataFrame."
        )

    if isinstance(nan_ratio, float):
        nan_ratio = [nan_ratio] * data_np.shape[1]
    elif isinstance(nan_ratio, list) and len(nan_ratio) != data_np.shape[1]:
        raise ValueError(
            "Length of nan_ratio list must match the number of columns in data."
        )

    for col in range(data_np.shape[1]):
        num_values = data_np.shape[0]
        num_nans = int(nan_ratio[col] * num_values)
        nan_indices = np.random.choice(num_values, num_nans, replace=False)
        data_np[nan_indices, col] = np.nan

    if is_tensor:
        return torch.tensor(data_np)
    else:
        return pd.DataFrame(data_np, columns=data.columns)


def impute_nans(X_train, X_valid=None, X_test=None):
    X_train_imp = X_train.clone()
    X_valid_imp = X_valid.clone() if X_valid is not None else None
    X_test_imp = X_test.clone() if X_test is not None else None

    # Impute NaNs with the mean of the available values in the training set
    nan_mask = torch.isnan(X_train)
    mean_values = torch.nanmean(X_train, dim=0)
    mean_values[torch.isnan(mean_values)] = (
        0  # Replace NaNs with 0 if all values are missing
    )
    X_train_imp[nan_mask] = mean_values[nan_mask.nonzero(as_tuple=True)[1]]

    # X_train_imp[nan_mask] =  torch.rand(X_train_imp[nan_mask].shape)
    nan_mask_test = torch.isnan(X_test) if X_test is not None else None
    nan_mask_valid = torch.isnan(X_valid) if X_valid is not None else None
    if X_test is not None:
        X_test_imp[nan_mask_test] = mean_values[nan_mask_test.nonzero(as_tuple=True)[1]]
        # X_test_imp[nan_mask_test] = torch.rand(X_test_imp[nan_mask_test].shape)
    if X_valid is not None:
        X_valid_imp[nan_mask_valid] = mean_values[
            nan_mask_valid.nonzero(as_tuple=True)[1]
        ]
        # X_valid_imp[nan_mask_valid] = torch.rand(X_valid_imp[nan_mask_valid].shape)
    return X_train_imp, X_valid_imp, X_test_imp, nan_mask, nan_mask_valid, nan_mask_test


def generate_y(
    X,
    dependency_type="complex",
    output_type="binary",
    threshold=0.7,
):
    """
    Computes target y based on either a simple or complex dependency on input features in X.

    Parameters:
    - X (torch.Tensor): Input tensor of shape (n_samples, n_features) where each column corresponds to a feature.
    - dependency_type (str): Specifies the dependency type; either "simple" or "complex".
    - threshold (float): Threshold value for binary classification.

    Returns:
    - y (torch.Tensor): target tensor of shape (n_samples, 1)
    """
    num_features = X.size(1)

    if dependency_type == "simple":
        # Simple dependency: Linear combination with random coefficients and a noise term
        coefficients = (
            torch.rand(num_features) * 0.6 - 0.3
        )  # Coefficients in range [-0.3, 0.3]
        y_continuous = (X @ coefficients + 0.1 * torch.randn(X.size(0))).view(-1, 1)

    elif dependency_type == "complex":
        # Complex dependency: Nonlinear transformations and interactions between features
        y_continuous = torch.zeros(X.size(0), 1)

        # Add a sine transformation for each feature and weighted sum with noise
        for i in range(num_features):
            y_continuous += torch.sin(X[:, i]).view(-1, 1) * (0.2 * torch.randn(1))

        # Add squared terms for each feature
        for i in range(num_features):
            y_continuous += 0.5 * X[:, i].view(-1, 1) ** 2

        # Add interactions between pairs of features
        for i in range(num_features):
            for j in range(i + 1, num_features):
                interaction_term = X[:, i] * X[:, j]
                y_continuous += 0.1 * interaction_term.view(-1, 1)

        # Add higher order terms for each feature
        for i in range(num_features):
            y_continuous += 0.05 * X[:, i].view(-1, 1) ** 3

        # Include additional random noise
        y_continuous += 0.1 * torch.randn(X.size(0), 1)
    else:
        raise ValueError("dependency_type must be either 'simple' or 'complex'.")

    # Convert continuous target to binary using the threshold
    y_binary = (y_continuous > threshold).float()
    if output_type == "binary":
        return y_binary, X
    else:
        return y_continuous, X


def select_features(X, avail_features, all_features):
    "select the available features for each node"
    not_feature_indices = [
        i for i, feature in enumerate(all_features) if feature not in avail_features
    ]
    if isinstance(X, torch.Tensor):
        X[:, not_feature_indices] = torch.tensor(float("nan"))
    else:  # Assume Pandas DataFrame
        X.loc[
            :, [feature for feature in all_features if feature not in avail_features]
        ] = np.nan

    return X


def preprocess_data_for_training_shift(
    node_data, binarize_feats, all_features, node_feature_sets, impute, nan_ratio, seed
):
    "generate synthetic data for each node"
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    node_data_train = []
    node_data_test = []
    node_data_valid = []
    node_data_train_imp = []
    node_data_test_imp = []
    node_data_valid_imp = []

    for i, features in enumerate(node_feature_sets):
        X_data = node_data[i][0]
        y_data = node_data[i][1]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_data, y_data, test_size=0.6, random_state=42
        )

        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,  # half of temp => 20% of total
            random_state=42,
        )

        # 3) Convert numpy arrays to torch tensors
        X_full = torch.tensor(X_train, dtype=torch.float32)
        X_full_valid = torch.tensor(X_valid, dtype=torch.float32)
        X_full_test = torch.tensor(X_test, dtype=torch.float32)

        # Reshape y to have a single column
        y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_valid = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        # 1) Local feature scaling
        # -----------------------------
        mean_i = X_full.mean(axis=0)
        std_i = X_full.std(axis=0)
        # Avoid division by zero
        std_i[std_i < 1e-12] = 1e-12

        cols_to_scale = [i for i in range(X_full.shape[1]) if i not in binarize_feats]
        X_full_scaled = X_full.clone()

        X_full_scaled[:, cols_to_scale] = (
            X_full[:, cols_to_scale] - mean_i[cols_to_scale]
        ) / std_i[cols_to_scale]
        X_valid_scaled = X_full_valid.clone()
        X_valid_scaled[:, cols_to_scale] = (
            X_full_valid[:, cols_to_scale] - mean_i[cols_to_scale]
        ) / std_i[cols_to_scale]
        X_test_scaled = X_full_test.clone()
        X_test_scaled[:, cols_to_scale] = (
            X_full_test[:, cols_to_scale] - mean_i[cols_to_scale]
        ) / std_i[cols_to_scale]

        # select available features for each node
        X_node = select_features(
            X_full,
            features,
            all_features,
        )
        X_node_valid = select_features(
            X_full_valid,
            features,
            all_features,
        )
        X_node_test = select_features(
            X_full_test,
            features,
            all_features,
        )

        avail_features = features if not impute else all_features
        # introduce some missingness
        nan_ratio_ = nan_ratio[i]
        X_node = introduce_nans(X_node, nan_ratio=nan_ratio_)
        X_node_valid = introduce_nans(X_node_valid, nan_ratio=nan_ratio_)
        X_node_test = introduce_nans(X_node_test, nan_ratio=nan_ratio_)

        node_data_train.append(
            (X_node, y, avail_features, torch.isnan(X_node), torch.isnan(y))
        )
        node_data_valid.append(
            (
                X_node_valid,
                y_valid,
                avail_features,
                torch.isnan(X_node_valid),
                torch.isnan(y_valid),
            )
        )
        node_data_test.append(
            (
                X_node_test,
                y_test,
                avail_features,
                torch.isnan(X_node_test),
                torch.isnan(y_test),
            )
        )

    X_centralized_train = torch.cat(
        [data[0].clone().detach() for data in node_data_train]
    )
    X_centralized_valid = torch.cat(
        [data[0].clone().detach() for data in node_data_valid]
    )
    # impute missing data for baseline
    if impute:
        for i, (X_train, y_train, features, _, _) in enumerate(node_data_train):
            (
                X_train_imp,
                X_valid_imp,
                X_test_imp,
                nan_mask,
                nan_mask_valid,
                nan_mask_test,
            ) = impute_nans(X_train, node_data_valid[i][0], node_data_test[i][0])
            node_data_train_imp.append(
                (X_train_imp, y_train, features, nan_mask, torch.isnan(y_train))
            )
            node_data_test_imp.append(
                (
                    X_test_imp,
                    node_data_test[i][1],
                    features,
                    nan_mask_test,
                    torch.isnan(node_data_test[i][1]),
                )
            )
            node_data_valid_imp.append(
                (
                    X_valid_imp,
                    node_data_valid[i][1],
                    features,
                    nan_mask_valid,
                    torch.isnan(node_data_valid[i][1]),
                )
            )
        X_centralized_train, X_centralized_valid, _, _, _, _ = impute_nans(
            X_centralized_train, X_centralized_valid
        )
    else:
        node_data_train_imp = node_data_train
        node_data_valid_imp = node_data_valid
        node_data_test_imp = node_data_test

    y_centralized_train = torch.cat([data[1] for data in node_data_train_imp], dim=0)
    y_centralized_valid = torch.cat([data[1] for data in node_data_valid_imp], dim=0)

    return (
        X_centralized_train,
        X_centralized_valid,
        y_centralized_train,
        y_centralized_valid,
        node_data_train_imp,
        node_data_valid_imp,
        node_data_test_imp,
    )
