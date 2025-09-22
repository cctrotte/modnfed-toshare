import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import average_precision_score
import numpy as np


class FeatureEncoder(nn.Module):
    "feature specific encoder module"

    def __init__(self, input_dim, state_dim, hidden_layers=[16, 16]):
        super(FeatureEncoder, self).__init__()
        layer_dims = [input_dim + state_dim] + hidden_layers + [state_dim]
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # Apply activation only for intermediate layers
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, feature, state):
        x = torch.cat((feature, state), dim=1)
        return self.network(x)





class FeatureDecoder(nn.Module):
    "feature specific decoder module (not used atm, not sure to keep it)"

    def __init__(self, state_dim, output_dim, hidden_layers=[16, 16]):
        super(FeatureDecoder, self).__init__()
        layer_dims = [state_dim] + hidden_layers + [output_dim]
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # Apply activation only for intermediate layers
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class TargetDecoder(nn.Module):
    "target decoder module"

    def __init__(
        self, state_dim, output_dim, hidden_layers=[16, 16], output_type="continuous"
    ):
        """
        Initializes the TargetDecoder model with either continuous or binary output.

        Parameters:
        - state_dim (int): Dimension of the state.
        - output_dim (int): Dimension of the output.
        - hidden_layers (list of int): List containing the sizes of the hidden layers.
        - output_type (str): Specifies the output type; either "continuous" or "binary".
        """
        super(TargetDecoder, self).__init__()
        self.output_type = output_type
        layer_dims = [state_dim] + hidden_layers + [output_dim]
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # Apply activation only for intermediate layers
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        output = self.network(state)
        # if self.output_type == "binary":
        #     # Apply sigmoid activation for binary classification
        #     output = torch.sigmoid(output)
        return output



class ModularModel(nn.Module):
    "MoDN model"

    def __init__(
        self,
        feature_names,
        target_names_and_types,
        state_dim,
        hidden_layers_enc=[16],
        hidden_layers_enc_private=[],
        hidden_layers_dec=[16],
        hidden_layers_dec_private=[],
        hidden_layers_feat_dec=[],
        feat_decoding=False,
        # output_type="continuous",
        predict_all=True,
        shuffle=True,
        input_dims=None,
        output_dims=None,
        node_id = None,
    ):
        super(ModularModel, self).__init__()
        self.state_dim = state_dim
        self.feature_names = sorted(feature_names)
        self.target_names = sorted(list(target_names_and_types.keys()))
        self.target_names_and_types = target_names_and_types
        # self.output_type = output_type
        self.feat_decoding = feat_decoding
        self.shuffle = shuffle
        self.predict_all = predict_all
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.node_id = node_id

        # Create enocders with or without private personalization layers
        if len(hidden_layers_enc_private) == 0:
            self.encoders = nn.ModuleDict(
                {
                    name: FeatureEncoder(
                        len(self.input_dims[name]), state_dim, hidden_layers_enc
                    )
                    for name in self.feature_names
                }
            )
        else:
            self.encoders = nn.ModuleDict(
                {
                    name: FeatureEncoderShift(
                        len(self.input_dims[name]),
                        state_dim,
                        hidden_layers_enc,
                        hidden_layers_enc_private,
                    )
                    for name in self.feature_names
                }
            )
        # TODO implement input_dims for feature decoding
        if self.feat_decoding:
            self.decoders = nn.ModuleDict(
                {
                    name: FeatureDecoder(state_dim, 1, hidden_layers_feat_dec)
                    for name in self.feature_names
                }
            )
        
        # target decoders with or with personalization layers
        if len(hidden_layers_dec_private) == 0:
            self.target_decoders = nn.ModuleDict(
                {
                    name: TargetDecoder(
                        state_dim,
                        1,
                        hidden_layers_dec,
                        self.target_names_and_types[name],
                    )
                    for name in self.target_names
                }
            )
        else:
            self.target_decoders = nn.ModuleDict(
                {
                    name: TargetDecoderShift(
                        state_dim,
                        1,
                        hidden_layers_dec,
                        hidden_layers_dec_private,
                        self.target_names_and_types[name],
                    )
                    for name in self.target_names
                }
            )

    def apply_encoder_by_name(self, state, feature, feature_name):
        """Applies the encoder for the specified feature to update the state."""
        if torch.isnan(feature).any().item():
            raise ValueError("Trying to encode a feature with NaN values")
        return self.encoders[feature_name](feature, state)
    

    def encode_all_features(self, x):
        """Encodes all features sequentially to update the state."""
        availability_mask = ~torch.isnan(x)

        batch_size = x.size(0)
        state = torch.zeros(batch_size, self.state_dim)  # initial state

        # Update state with each feature encoder in a fixed order
        for feature_name in self.feature_names:
            mask = availability_mask[:, self.input_dims[feature_name]].all(dim=1)

            feature = x[:, self.input_dims[feature_name]]
            state[mask] += self.apply_encoder_by_name(
                state[mask], feature[mask], feature_name
            )

        return state

    def forward(self, x, feat_order = None):
        """applies encoders and decoders for standard training."""
        batch_size = x.size(0)
        availability_mask = ~torch.isnan(x)
        state = torch.zeros(batch_size, self.state_dim)
        reconstructions = {
            name: None for name in self.feature_names
        }  # placeholders for reconstructions
        feature_indices = list(range(len(self.feature_names)))
        # randomly shuffle order in which encoders are applied
        if self.shuffle and feat_order is None:
            random.shuffle(feature_indices)
        elif feat_order is not None:
            feature_indices = feat_order
        target_prediction = torch.empty(
            len(feature_indices)+1, batch_size, len(self.target_names)
        )
        # predict before encoding any feature
        if self.predict_all:
            for j, target_name in enumerate(self.target_names):
                target_prediction[0, :, j] = self.target_decoders[target_name](
                    state
                ).flatten()
        for ix, i in enumerate(feature_indices, start = 1):
            feature_name = self.feature_names[i]
            feature = x[:, self.input_dims[feature_name]]
            mask = availability_mask[:, self.input_dims[feature_name]].all(dim=1)
            # is there a cleaner way?
            next_state = state.clone()
            next_state[mask] += self.encoders[feature_name](
                feature[mask], next_state[mask]
            )
            state = next_state

            if self.feat_decoding:
                for name in self.feature_names:
                    # Randomly decide whether to decode any feature at this point
                    if reconstructions[name] is None and random.random() < 0.5:
                        reconstructions[name] = self.decoders[name](state)

            # predict the target each time a new feature is encoded
            if self.predict_all:
                for j, target_name in enumerate(self.target_names):
                    target_prediction[ix, :, j] = self.target_decoders[target_name](
                        state
                    ).flatten()
                # target_prediction[ix, :] = self.target_decoder(state)
        if not self.predict_all:
            target_prediction = torch.empty(batch_size, len(self.target_names))
            for j, target_name in enumerate(self.target_names):
                target_prediction[:, j] = self.target_decoders[target_name](
                    state
                ).flatten()
            # target_prediction = self.target_decoder(state)
        # Ensure all features are decoded by the end
        if self.feat_decoding:
            for name in self.feature_names:
                if reconstructions[name] is None:
                    reconstructions[name] = self.decoders[name](state)

        # Concatenate reconstructions in the same order as feature_names
        reconstructed_features = (
            torch.cat([reconstructions[name] for name in self.feature_names], dim=1)
            if self.feat_decoding
            else None
        )

        return reconstructed_features, target_prediction, feature_indices

    def compute_loss_after_encoding(
        self,
        x,
        y,
        feature_names,
        target_names,
        nan_mask,
        nan_target_mask,
        criterion_targets,
    ):
        """Computes the loss after applying each encoder in sequence."""

        criterion_features = nn.MSELoss()

        batch_size = x.size(0)
        availability_mask = ~nan_mask

        state = torch.zeros(batch_size, self.state_dim)  # initial state

        # Store intermediate losses
        losses = []
        f1_scores = []

        # Calculate loss with no encoder applied
        reconstructed_features = (
            torch.cat([self.decoders[name](state) for name in feature_names], dim=1)
            if self.feat_decoding
            else None
        )
        target_prediction = torch.empty(batch_size, len(target_names))
        for j, target_name in enumerate(target_names):
            target_prediction[:, j] = self.target_decoders[target_name](state).flatten()
        # target_prediction = self.target_decoder(state)

        loss_features = torch.tensor(0.0, dtype=torch.float32)
        if self.feat_decoding:
            # TODO adapt this to catogirucal features and input dims if we keep it
            for i, feature_name in enumerate(self.feature_names):
                feature = x[:, i : i + 1]
                reconstruction = reconstructed_features[:, i : i + 1]
                mask = availability_mask[:, i : i + 1]  # Mask for this feature

                # Only compute loss where feature is available
                if mask.any():
                    loss_features += criterion_features(
                        reconstruction[mask], feature[mask]
                    )
        loss_targets = []
        f1_scores_targets = []
        # prediction before providing any information
        for j, target_name in enumerate(target_names):
            target = y[:, self.output_dims[target_name]]
            avail_mask = ~nan_target_mask[:, self.output_dims[target_name]]

            loss_targets.append(
                criterion_targets[target_name](
                    target_prediction[avail_mask, j], target[avail_mask].flatten()
                ).item()
            )
            if self.target_names_and_types[target_name] == "binary":
                # Apply sigmoid activation for binary classification
                y_pred_class = (torch.sigmoid(target_prediction[:, j]) >= 0.5).float()
                f1_scores_targets.append(
                    f1_score(
                        target[avail_mask].numpy(),
                        y_pred_class[avail_mask].numpy(),
                        average="macro",
                    )
                )
        losses.append((0, loss_features.item(), torch.mean(torch.tensor(loss_targets))))
        f1_scores.append((0, torch.mean(torch.tensor(f1_scores_targets))))

        # Apply encoders sequentially and calculate loss after each
        for i, feature_name in enumerate(feature_names, start=1):

            mask = availability_mask[:, self.input_dims[feature_name]].all(dim=1)
            feature = x[:, self.input_dims[feature_name]]

            state[mask] += self.apply_encoder_by_name(
                state[mask], feature[mask], feature_name
            )

            # Compute reconstruction and target loss after each encoder application
            reconstructed_features = (
                torch.cat([self.decoders[name](state) for name in feature_names], dim=1)
                if self.feat_decoding
                else None
            )
            # target prediction
            target_prediction = torch.empty(batch_size, len(target_names))
            for j, target_name in enumerate(target_names):
                target_prediction[:, j] = self.target_decoders[target_name](
                    state
                ).flatten()
            loss_features = torch.tensor(0, dtype=torch.float32)
            if self.feat_decoding:
                # TODO adapt this to catogirucal features and input dims if we keep it
                for j, feature_name_d in enumerate(self.feature_names):
                    feature = x[:, j : j + 1]
                    reconstruction = reconstructed_features[:, j : j + 1]
                    mask = availability_mask[
                        :, j : j + 1
                    ].flatten()  # Mask for this feature

                    # Only compute loss where feature is available
                    if mask.any():
                        loss_features += criterion_features(
                            reconstruction[mask], feature[mask]
                        )
            # average losses
            loss_targets = []
            f1_scores_targets = []
            # also store per target final loss
            loss_per_target = {}
            f1_per_target = {}

            # compute target loss
            for j, target_name in enumerate(target_names):
                target = y[:, self.output_dims[target_name]]
                avail_mask = ~nan_target_mask[:, self.output_dims[target_name]]

                loss_targets.append(
                    criterion_targets[target_name](
                        target_prediction[avail_mask, j], target[avail_mask].flatten()
                    ).item()
                )
                loss_per_target[target_name] = criterion_targets[target_name](
                    target_prediction[avail_mask, j], target[avail_mask].flatten()
                ).item()
                if self.target_names_and_types[target_name] == "binary":
                    # Apply sigmoid activation for binary classification
                    y_pred_class = (
                        torch.sigmoid(target_prediction[:, j]) >= 0.5
                    ).float()
                    f1_scores_targets.append(
                        f1_score(
                            target[avail_mask].numpy(),
                            y_pred_class[avail_mask].numpy(),
                            average="macro",
                        )
                    )
                    f1_per_target[target_name] = f1_score(
                        target[avail_mask].numpy(),
                        y_pred_class[avail_mask].numpy(),
                        average="macro",
                    )
            losses.append(
                (i, loss_features.item(), torch.mean(torch.tensor(loss_targets)))
            )
            f1_scores.append((i, torch.mean(torch.tensor(f1_scores_targets))))

        return losses, f1_scores

    def compute_final_loss(
        self, x, y, target_names, nan_target_mask, criterion_targets
    ):
        """Computes the loss after encoding all features."""

        batch_size = x.size(0)
        state = self.encode_all_features(x)
        # target prediction
        target_prediction = torch.empty(batch_size, len(target_names))
        for j, target_name in enumerate(target_names):
            target_prediction[:, j] = self.target_decoders[target_name](state).flatten()

        # also store per target final loss
        loss_per_target = {}
        targets = {
            name: {"prediction": np.nan, "ground_truth": np.nan, "scores": np.nan}
            for name in target_names
        }

        # compute target loss
        for j, target_name in enumerate(target_names):
            target = y[:, self.output_dims[target_name]]
            avail_mask = ~nan_target_mask[:, self.output_dims[target_name]]

            loss_per_target[target_name] = criterion_targets[target_name](
                target_prediction[avail_mask, j], target[avail_mask].flatten()
            ).item()
            if self.target_names_and_types[target_name] == "binary":
                # Apply sigmoid activation for binary classification
                y_pred_class = (torch.sigmoid(target_prediction[:, j]) >= 0.5).float()
                targets[target_name]["prediction"] = y_pred_class[avail_mask].numpy()
                targets[target_name]["ground_truth"] = target[avail_mask].numpy()
                targets[target_name]["scores"] = (
                    target_prediction[avail_mask, j].detach().numpy()
                )
                # f1_per_target[target_name] = f1_score(
                #     target[avail_mask].numpy(),
                #     y_pred_class[avail_mask].numpy(),
                #     average="macro",
                # )
                # auprc_per_target[target_name] = average_precision_score(
                #     target[avail_mask].numpy(), y_pred_class[avail_mask].numpy()
                # )

        return loss_per_target, targets



class FeatureEncoderShift(nn.Module):
    """
    A feature encoder with both a shared (federated) part and a private (local-only) part.
    NOT REALLY USED ANYMORE
    """

    def __init__(
        self,
        input_dim,
        state_dim,
        hidden_layers_shared=[16],
        hidden_layers_private=[16],
    ):
        super(FeatureEncoderShift, self).__init__()

        if len(hidden_layers_private) == 0:
            # 1) Make a single sub-network that goes:
            #    (input_dim + state_dim) -> hidden_layers_shared... -> state_dim
            # 2) Then set self.private_encoder to nn.Identity()
            layer_dims_shared = (
                [input_dim + state_dim] + hidden_layers_shared + [state_dim]
            )
            layers_shared = []
            for i in range(len(layer_dims_shared) - 1):
                layers_shared.append(
                    nn.Linear(layer_dims_shared[i], layer_dims_shared[i + 1])
                )
                if i < len(layer_dims_shared) - 2:
                    layers_shared.append(nn.ReLU())
            self.shared_encoder = nn.Sequential(*layers_shared)

            # The private encoder is just identity (no parameters)
            self.private_encoder = nn.Identity()
        elif len(hidden_layers_shared) == 0:

            self.shared_encoder = nn.Identity()
            layer_dims_private = (
                [input_dim + state_dim] + hidden_layers_private + [state_dim]
            )
            layers_private = []
            for i in range(len(layer_dims_private) - 1):
                layers_private.append(
                    nn.Linear(layer_dims_private[i], layer_dims_private[i + 1])
                )
                if i < len(layer_dims_private) - 1:
                    layers_private.append(nn.ReLU())
            self.private_encoder = nn.Sequential(*layers_private)

        else:
            # 1) Build the shared sub-network
            layer_dims_shared = [input_dim + state_dim] + hidden_layers_shared
            layers_shared = []
            for i in range(len(layer_dims_shared) - 1):
                layers_shared.append(
                    nn.Linear(layer_dims_shared[i], layer_dims_shared[i + 1])
                )
                layers_shared.append(nn.ReLU())
            self.shared_encoder = nn.Sequential(*layers_shared)

            # 2) Build the private sub-network
            layer_dims_private = (
                [layer_dims_shared[-1]] + hidden_layers_private + [state_dim]
            )
            layers_private = []
            for i in range(len(layer_dims_private) - 1):
                layers_private.append(
                    nn.Linear(layer_dims_private[i], layer_dims_private[i + 1])
                )
                if i < len(layer_dims_private) - 1:
                    layers_private.append(nn.ReLU())
            self.private_encoder = nn.Sequential(*layers_private)

    def forward(self, feature, state):
        """
        Combine feature and state, pass them through the shared encoder, then the private one.
        """
        x = torch.cat((feature, state), dim=1)

        # Shared 
        x = self.shared_encoder(x)

        # Private 
        x = self.private_encoder(x)

        return x


class TargetDecoderShift(nn.Module):
    """
    A target decoder with both a shared (federated) submodule
    and a private (local-only) submodule.
    NOT REALLY USED ANYMORE
    """

    def __init__(
        self,
        state_dim,
        output_dim,
        hidden_layers_shared=[16],
        hidden_layers_private=[16],
        output_type="continuous",
    ):
        super(TargetDecoderShift, self).__init__()
        self.output_type = output_type

        # Case: No private layers at all
        if len(hidden_layers_private) == 0:
            # Combine hidden_layers_shared + [output_dim] to make a single shared module
            layer_dims_shared = [state_dim] + hidden_layers_shared + [output_dim]
            layers_shared = []
            for i in range(len(layer_dims_shared) - 1):
                layers_shared.append(
                    nn.Linear(layer_dims_shared[i], layer_dims_shared[i + 1])
                )
                # Apply ReLU to all but the final output layer
                if i < len(layer_dims_shared) - 2:
                    layers_shared.append(nn.ReLU())
            self.shared_decoder = nn.Sequential(*layers_shared)

            # No private decoder
            self.private_decoder = nn.Identity()

        elif len(hidden_layers_shared) == 0:
            self.shared_decoder = nn.Identity()

            layer_dims_private = [state_dim] + hidden_layers_private + [output_dim]
            layers_private = []
            for i in range(len(layer_dims_private) - 1):
                layers_private.append(
                    nn.Linear(layer_dims_private[i], layer_dims_private[i + 1])
                )
                # ReLU for hidden layers, skip for final output layer
                if i < len(layer_dims_private) - 2:
                    layers_private.append(nn.ReLU())

            self.private_decoder = nn.Sequential(*layers_private)

        else:
            # -------------------------
            # 1) Shared decoder submodule
            # -------------------------
            layer_dims_shared = [state_dim] + hidden_layers_shared
            layers_shared = []
            for i in range(len(layer_dims_shared) - 1):
                layers_shared.append(
                    nn.Linear(layer_dims_shared[i], layer_dims_shared[i + 1])
                )
                layers_shared.append(nn.ReLU())
            self.shared_decoder = nn.Sequential(*layers_shared)

            # -------------------------
            # 2) Private decoder submodule
            # -------------------------
            # Maps from the last shared dimension to output_dim
            layer_dims_private = (
                [layer_dims_shared[-1]] + hidden_layers_private + [output_dim]
            )
            layers_private = []
            for i in range(len(layer_dims_private) - 1):
                layers_private.append(
                    nn.Linear(layer_dims_private[i], layer_dims_private[i + 1])
                )
                # ReLU for hidden layers, skip for final output layer
                if i < len(layer_dims_private) - 2:
                    layers_private.append(nn.ReLU())

            self.private_decoder = nn.Sequential(*layers_private)

    def forward(self, state):
        """
        Pass `state` through the shared decoder, then
        the private decoder.
        """
        # Shared forward pass
        x = self.shared_decoder(state)
        # Private forward pass
        output = self.private_decoder(x)

        return output