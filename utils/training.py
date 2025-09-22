import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


def train_on_node(
    model,
    global_model,
    X,
    y,
    criterion_features,
    criterion_targets,
    local_epochs,
    feature_names,
    target_names,
    nan_mask,
    nan_target_mask,
    batch_size,
    learning_rate,
    feature_decoding,
    mu = 0,
):

    dataset = TensorDataset(X, y, nan_mask, nan_target_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(local_epochs):

        for X_batch, y_batch, nan_mask_batch, nan_target_mask_batch in dataloader:
            optimizer.zero_grad()

            if model.predict_all:
                # adapt size of target since the model predicts each time a new feature is encoded

                y_stacked = y_batch.repeat(len(feature_names) + 1, 1, 1)
            else:
                y_stacked = y_batch

            # Run forward pass based only on available features
            reconstructed_features, predicted_targets,_ = model(X_batch)
            availability_mask = ~nan_mask_batch

            # if predicted_targets.shape != y_stacked.shape:
            #     raise ValueError(
            #         f"Shape mismatch: predicted shape is {predicted_targets.shape}, "
            #         f"but target shape is {y_stacked.shape}"
            #     )

            # Compute losses
            loss_features = torch.tensor(0.0, device=X_batch.device)
            if feature_decoding:
                for i, feature_name in enumerate(feature_names):
                    feature = X_batch[:, i : i + 1]
                    reconstruction = reconstructed_features[:, i : i + 1]
                    mask = availability_mask[:, i : i + 1].flatten()

                    # Only compute loss where feature is available
                    if mask.any():
                        loss_features += criterion_features(
                            reconstruction[mask], feature[mask]
                        )
            loss = loss_features
            for j, target_name in enumerate(target_names):
                avail_mask = ~nan_target_mask_batch[:, model.output_dims[target_name]]
                target = y_stacked[..., model.output_dims[target_name]]
                prediction = predicted_targets[..., j]
                loss_targets = criterion_targets[target_name](
                    prediction[..., avail_mask], target[..., avail_mask]
                )
                loss += loss_targets
            # FedProx Proximal Term (same as FedAVG for mu=0)
            prox_term = torch.tensor(0.0, device=X.device)
            # grab named parameters from both models
            local_params  = dict(model.named_parameters())
            global_params = dict(global_model.named_parameters())

            # only intersect on keys they both share
            for name in set(local_params.keys()) & set(global_params.keys()):
                local_w  = local_params[name]
                global_w = global_params[name]
                prox_term += torch.norm(local_w - global_w, p=2)**2
            prox_term = (mu / 2) * prox_term # scale by mu/2
            loss += prox_term

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()



def aggregate_models(
    global_model,
    local_models,
    sample_counts,
    weight=True,
    model_type="modn",
):
    global_state_dict = global_model.state_dict()
    total_samples = sum(sample_counts)

    if model_type == "modn":

        # Only aggregate weights for shared features
        for key in global_state_dict.keys():
            if "private" in key:
                continue
            is_shared = any(key in model.state_dict() for model in local_models)
            if is_shared:
                if weight:
                    weighted_sum = sum(
                        (
                            local_model.state_dict()[key]
                            * (sample_counts[i] / total_samples)
                        )
                        for i, local_model in enumerate(local_models)
                        if key in local_model.state_dict()  # Ensure key exists
                    )
                    global_state_dict[key] = weighted_sum
                else:
                    relevant_models = [
                        model for model in local_models if key in model.state_dict()
                    ]
                    global_state_dict[key] = torch.mean(
                        torch.stack(
                            [model.state_dict()[key] for model in relevant_models]
                        ),
                        dim=0,
                    )
    else:
        for key in global_state_dict.keys():
            if not all(key in model.state_dict() for model in local_models):
                # for the mlp baseline we assume that all models have the same keys
                raise KeyError(
                    f"Parameter '{key}' is missing in at least one local model. Aggregation cannot proceed."
                )
            if weight:
                weighted_sum = sum(
                    local_model.state_dict()[key] * (sample_counts[i] / total_samples)
                    for i, local_model in enumerate(local_models)
                )
                global_state_dict[key] = weighted_sum
            else:
                global_state_dict[key] = torch.mean(
                    torch.stack(
                        [local_model.state_dict()[key] for local_model in local_models]
                    ),
                    dim=0,
                )

    global_model.load_state_dict(global_state_dict)




