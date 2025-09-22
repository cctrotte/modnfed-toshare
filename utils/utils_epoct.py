import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from utils.data import introduce_nans


def get_class_weights(
    weight_targets, y_centralized_train, node_data_train, all_targets, node_list
):
    if weight_targets:
        target_weights = {
            node: {target: None for target in all_targets} for node in node_list
        }
        target_weights["centralized"] = {target: None for target in all_targets}

        for j, target in enumerate(all_targets):
            for ix, node in enumerate(node_list):
                targets = np.array(node_data_train[ix][1][:, j])
                targets = targets[~np.isnan(targets)]
                if len(targets) == 0:
                    pos_weight = None
                elif len(np.unique(targets)) == 1:
                    pos_weight = None
                else:
                    num_pos = targets.sum()
                    num_neg = len(targets) - num_pos
                    pos_weight = torch.tensor([num_neg / num_pos])
                target_weights[node][target] = pos_weight

            targets_centralized = np.array(y_centralized_train[:, j])
            targets_centralized = targets_centralized[~np.isnan(targets_centralized)]
            num_pos_centralized = targets_centralized.sum()
            num_neg_centralized = len(targets_centralized) - num_pos_centralized
            pos_weight_centralized = torch.tensor(
                [num_neg_centralized / num_pos_centralized]
            )
            if len(np.unique(targets_centralized)) == 1:
                pos_weight_centralized = None
            else:
                target_weights["centralized"][target] = pos_weight_centralized
    else:
        target_weights = {
            node: {target: None for target in all_targets} for node in node_list
        }
        target_weights["centralized"] = {target: None for target in all_targets}
    return target_weights


def get_numeric_data(df, feature_types, target_types):
    df_c = df.copy()
    for feature, f_type in feature_types.items():
        if f_type == "binary":
            df_c[feature] = df_c[feature].replace(
                {
                    "Yes": 1.0,
                    "No": 0.0,
                    "Male": 0.0,
                    "Female": 1.0,
                    "epoct_plus_rwanda": 0.0,
                    "epoct_plus_tanzania": 1.0,
                }
            )
            df_c[feature] = df_c[feature].astype(float)
    for target, t_type in target_types.items():
        if t_type == "binary":
            df_c[target] = df_c[target].replace({"Yes": 1.0, "No": 0.0})
            df_c[target] = df_c[target].astype(float)

    return df_c


def get_feature_mappings(all_features, categorical_features, pipeline):

    # store a dict to map the feature names to the indices in the transformed array
    feature_index_mapping = {feat: [] for feat in all_features}
    transformed_feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    for idx, name in enumerate(transformed_feature_names):
        # Determine the original feature name based on the transformer prefix.
        if name.startswith("cont__"):
            orig_feat = name[len("cont__") :]
        elif name.startswith("bin__"):
            orig_feat = name[len("bin__") :]
        elif name.startswith("cat__"):
            # For categorical features, match using the original feature names.
            orig_feat = None
            for feat in categorical_features:
                # The categorical names are usually of the form "cat__{feat}_{category}"
                prefix = f"cat__{feat}_"
                if name.startswith(prefix):
                    orig_feat = feat
                    break
        else:
            # Fallback: if no prefix exists, assume the name is the original feature.
            orig_feat = name

        # Append the index to the list for that feature.
        feature_index_mapping.setdefault(orig_feat, []).append(idx)

    return feature_index_mapping


def get_data_for_nodes(
    all_features,
    all_targets,
    feature_types,
    target_types,
    node_variables,
    df_num,
    seed,
    model_type,
    percentage =1,
    split_ids = None,
):
    # target_types ununsed for now ,adapt if we have a continuous target
    node_data_train, node_data_valid, node_data_test = [], [], []

    n_samples = []

    # needed to store possible categories to ensure that the categories are the same across all nodes when one hot encoding to have same module architecture
    categorical_features = [
        feat for feat in all_features if feature_types.get(feat) == "categorical"
    ]
    categories_list = []
    for feat in categorical_features:
        cats = sorted(df_num[feat].dropna().unique())
        if "missing" not in cats:
            cats = ["missing"] + cats
        categories_list.append(cats)

    # also we temporarily use "missing" as a category for missing values (workaround to avoid an error)
    # later in the code (for modn at least) we replace all one-hot encoded columns back with np.nan if it was missing
    # TODO drop one column? Keep as is for now
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),  # Replace NaN with "missing"
            (
                "onehot",
                OneHotEncoder(handle_unknown="error", categories=categories_list),
            ),
        ]
    )
    for index, (node, data) in enumerate(node_variables.items()):
        X_full = df_num[df_num["health_facility_id"] == data["health_facility_id"]]
        missing_features = list(set(all_features) - set(data["features"]))
        missing_targets = list(set(all_targets) - set(data["targets"]))
        

        if split_ids is None:
            # train/test split based on patients
            unique_patients = X_full["patient_id"].unique()
            np.random.seed(seed)
            np.random.shuffle(unique_patients)
            # keep only a percentage of the data
            n_patients = max(int(len(unique_patients) * percentage), 200)

            unique_patients = unique_patients[:n_patients]
            train_ratio = 0.5
            valid_ratio = 0.3  # remaining will be test

            n_patients = len(unique_patients)
            n_train, n_valid = int(n_patients * train_ratio), int(n_patients * valid_ratio)

            train_patients, valid_patients, test_patients = (
                unique_patients[:n_train],
                unique_patients[n_train : n_train + n_valid],
                unique_patients[n_train + n_valid :],
            )
        else:
            train_patients, valid_patients, test_patients = split_ids[data["health_facility_id"]]["train"], split_ids[data["health_facility_id"]]["valid"], split_ids[data["health_facility_id"]]["test"]

        # Subset X_full based on the patient_id splits
        train_df, valid_df, test_df = (
            X_full[X_full["patient_id"].isin(train_patients)],
            X_full[X_full["patient_id"].isin(valid_patients)],
            X_full[X_full["patient_id"].isin(test_patients)],
        )

        # For the current node, select features and targets
        X_node_train, X_node_valid, X_node_test = (
            train_df[all_features],
            valid_df[all_features],
            test_df[all_features],
        )

        y_train, y_valid, y_test = (
            train_df[all_targets],
            valid_df[all_targets],
            test_df[all_targets],
        )

        (
            X_node_train.loc[:, missing_features],
            X_node_valid.loc[:, missing_features],
            X_node_test.loc[:, missing_features],
        ) = (np.nan, np.nan, np.nan)

        (
            y_train.loc[:, missing_targets],
            y_valid.loc[:, missing_targets],
            y_test.loc[:, missing_targets],
        ) = (np.nan, np.nan, np.nan)

        
        # artificially introduce some nans
        # nan_ratio_test = [[0]*len(all_features) for _ in range(3)]
        # nan_ratio_test[0][all_features.index('hemoglobin')] = 0.8
        # nan_ratio_test[1][all_features.index('malaria_rdt_or_microscopy')] = 0.8

        # X_node_train = introduce_nans(X_node_train, nan_ratio=nan_ratio_train[index])
        # X_node_valid = introduce_nans(X_node_valid, nan_ratio=0.5)
        #X_node_test = introduce_nans(X_node_test, nan_ratio=0.3)


        for col in feature_types.keys():

            if col in categorical_features:
                X_node_train[col], X_node_valid[col], X_node_test[col] = (
                    X_node_train[col].astype("object"),
                    X_node_valid[col].astype("object"),
                    X_node_test[col].astype("object"),
                )
            elif col in all_features:
                X_node_train[col], X_node_valid[col], X_node_test[col] = (
                    X_node_train[col].astype("float"),
                    X_node_valid[col].astype("float"),
                    X_node_test[col].astype("float"),
                )

        # Split the selected features into groups based on the YAML mapping
        continuous_features = [
            feat for feat in all_features if feature_types.get(feat) == "continuous"
        ]
        binary_features = [
            feat for feat in all_features if feature_types.get(feat) == "binary"
        ]
        categorical_features = [
            feat for feat in all_features if feature_types.get(feat) == "categorical"
        ]

        # Create a ColumnTransformer to handle different types of features:
        if model_type == "modn":

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cont", StandardScaler(), continuous_features),
                    ("bin", "passthrough", binary_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

        else:

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cont",
                        Pipeline(
                            steps=[
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="mean", keep_empty_features=True
                                    ),
                                ),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        continuous_features,
                    ),
                    (
                        "bin",
                        SimpleImputer(strategy="constant", fill_value=0.0),
                        binary_features,
                    ),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

        # Integrate the ColumnTransformer into a pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

        # Fit and transform the node features
        X_node_train_processed = torch.tensor(
            pipeline.fit_transform(X_node_train), dtype=torch.float32
        )
        X_node_valid_processed = torch.tensor(
            pipeline.transform(X_node_valid), dtype=torch.float32
        )
        X_node_test_processed = torch.tensor(
            pipeline.transform(X_node_test), dtype=torch.float32
        )

        feature_index_mapping = get_feature_mappings(
            all_features, categorical_features, pipeline
        )
        target_index_mapping = {elem: i for i, elem in enumerate(all_targets)}

        (
            X_node_train_processed_cp,
            X_node_valid_processed_cp,
            X_node_test_processed_cp,
        ) = (
            X_node_train_processed.clone(),
            X_node_valid_processed.clone(),
            X_node_test_processed.clone(),
        )

        # Set the missing values in the transformed array to NaN again (at least do that for modn)
        for feat in categorical_features:
            # Get the missing mask: True where the original value was NaN.
            missing_mask_train, missing_mask_valid, missing_mask_test = (
                X_node_train[feat].isna().to_numpy(),
                X_node_valid[feat].isna().to_numpy(),
                X_node_test[feat].isna().to_numpy(),
            )
            # Get the indices in the transformed array corresponding to this feature.
            indices = feature_index_mapping.get(feat, [])
            missing_rows_train, missing_rows_valid, missing_rows_test = (
                np.where(missing_mask_train)[0],
                np.where(missing_mask_valid)[0],
                np.where(missing_mask_test)[0],
            )
            # Set the corresponding entries in the transformed array to NaN
            if missing_rows_train.size > 0 and indices:
                X_node_train_processed_cp[np.ix_(missing_rows_train, indices)] = np.nan
            if missing_rows_valid.size > 0 and indices:
                X_node_valid_processed_cp[np.ix_(missing_rows_valid, indices)] = np.nan
            if missing_rows_test.size > 0 and indices:
                X_node_test_processed_cp[np.ix_(missing_rows_test, indices)] = np.nan
            # if model is modn we want to keep missing data
            if model_type == "modn":
                (
                    X_node_train_processed,
                    X_node_valid_processed,
                    X_node_test_processed,
                ) = (
                    X_node_train_processed_cp.clone(),
                    X_node_valid_processed_cp.clone(),
                    X_node_test_processed_cp.clone(),
                )

        target_names = data["targets"] if model_type == "modn" else all_targets

        node_data_train.append(
            (
                X_node_train_processed,
                torch.tensor(y_train.values, dtype=torch.float32),
                data["features"],
                target_names,
                torch.isnan(X_node_train_processed_cp),
                torch.isnan(torch.tensor(y_train.values, dtype=torch.float32)),
                train_patients,
                data["health_facility_id"]
            )
        )
        node_data_valid.append(
            (
                X_node_valid_processed,
                torch.tensor(y_valid.values, dtype=torch.float32),
                data["features"],
                target_names,
                torch.isnan(X_node_valid_processed_cp),
                torch.isnan(torch.tensor(y_valid.values, dtype=torch.float32)),
                valid_patients,
                data["health_facility_id"]

            )
        )
        node_data_test.append(
            (
                X_node_test_processed,
                torch.tensor(y_test.values, dtype=torch.float32),
                data["features"],
                target_names,
                torch.isnan(X_node_test_processed_cp),
                torch.isnan(torch.tensor(y_test.values, dtype=torch.float32)),
                test_patients,
                data["health_facility_id"]
            )
        )

        n_samples.append(len(X_node_train))

    X_centralized_train, X_centralized_valid = torch.cat(
        [data[0].clone().detach() for data in node_data_train]
    ), torch.cat([data[0].clone().detach() for data in node_data_valid])
    y_centralized_train, y_centralized_valid = torch.cat(
        [data[1] for data in node_data_train], dim=0
    ), torch.cat([data[1] for data in node_data_valid], dim=0)

    return (
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
    )
