import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import wandb


class BaselineModel(nn.Module):
    "MLP baseline model for federated learning."

    def __init__(
        self,
        input_dim,
        state_dim,
        hidden_layers_enc=[16, 16],
        hidden_layers_dec=[16, 16],
        hidden_layers_decoding=[16, 16],
        feat_decoding=False,
        feature_names = None,
        target_names_and_types=None,
        # output_type="continuous",
        predict_all=False,
        output_dims=None,
        node_id = None
    ):
        super(BaselineModel, self).__init__()
        # self.output_type = output_type
        self.feat_decoding = feat_decoding
        # only to match structure, this isnt actually used in bsl model
        self.predict_all = predict_all
        self.feature_names = feature_names
        self.output_dims = output_dims
        self.output_size = len(output_dims)
        self.target_names_and_types = target_names_and_types
        self.node_id = node_id

        # encoder layers map input features to latent representation
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_layers_enc:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        # "bottleneck" layer to have similar restriction as MoDN
        encoder_layers.append(nn.Linear(in_dim, state_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # feature decoding layers to map latent state back to original input features
        if self.feat_decoding:
            decoder_layers = []
            in_dim = state_dim
            for h_dim in hidden_layers_decoding:
                decoder_layers.append(nn.Linear(in_dim, h_dim))
                decoder_layers.append(nn.ReLU())
                in_dim = h_dim
            decoder_layers.append(nn.Linear(in_dim, input_dim))
            self.feature_decoder = nn.Sequential(*decoder_layers)

        # Target Prediction Head: Predicts target from latent state
        target_layers = []
        in_dim = state_dim
        for h_dim in hidden_layers_dec:
            target_layers.append(nn.Linear(in_dim, h_dim))
            target_layers.append(nn.ReLU())
            in_dim = h_dim
        target_layers.append(nn.Linear(in_dim, self.output_size))  # Output dim
        self.target_decoder = nn.Sequential(*target_layers)

    def forward(self, x):
        # Encode input to latent representation
        state = self.encoder(x)
        # Decode latent representation to reconstruct features
        if self.feat_decoding:
            reconstructed_features = self.feature_decoder(state)
        else:
            reconstructed_features = None
        # Predict target from latent representation
        target_prediction = self.target_decoder(state)
        # if self.output_type == "binary":
        #     # Apply sigmoid activation for binary classification
        #     target_prediction = torch.sigmoid(target_prediction)

        return reconstructed_features, target_prediction, []

    def compute_loss_after_encoding(
        self,
        x,
        y,
        feature_names=None,
        target_names=None,
        nan_mask=None,
        nan_target_mask=None,
        criterion_targets=None,
    ):
        """Computes the reconstruction and target prediction losses after encoding all features."""
        criterion = nn.MSELoss()

        # Forward pass to get reconstructions and target prediction
        reconstructed_features, target_prediction = self(x)

        # Create availability mask to exclude missing features
        availability_mask = ~nan_mask

        # Compute feature reconstruction loss only for available features
        loss_features = torch.tensor(0, dtype=torch.float32)
        if self.feat_decoding:
            # TODO adapt with input feature size if we keep that
            for i in range(x.size(1)):
                feature = x[:, i : i + 1]
                reconstruction = reconstructed_features[:, i : i + 1]
                mask = availability_mask[:, i : i + 1]  # Mask for this feature

                # Only compute loss where feature is available
                if mask.any():
                    loss_features += criterion(reconstruction[mask], feature[mask])

        loss_targets = []
        f1_scores_targets = []
        # compute target loss
        for j, target_name in enumerate(target_names):
            target = y[:, self.output_dims[target_name]]
            avail_mask = ~nan_target_mask[:, self.output_dims[target_name]]

            if len(target[avail_mask]) > 0:
                loss_targets.append(
                    criterion_targets[target_name](
                        target_prediction[avail_mask, j], target[avail_mask].flatten()
                    ).item()
                )
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

        # Return feature and target losses
        return [(0, loss_features.item(), torch.mean(torch.tensor(loss_targets)))], [
            (0, torch.mean(torch.tensor(f1_scores_targets)))
        ]

    def compute_final_loss(
        self,
        x,
        y,
        target_names,
        nan_target_mask,
        criterion_targets,
    ):
        """Computes the reconstruction and target prediction losses after encoding all features."""
        # Forward pass to get reconstructions and target prediction
        _, target_prediction, _ = self(x)

        loss_per_target = {}
        targets = {
            name: {"prediction": np.nan, "ground_truth": np.nan, "scores": np.nan}
            for name in target_names
        }

        # compute target loss
        for j, target_name in enumerate(target_names):
            target = y[:, self.output_dims[target_name]]
            avail_mask = ~nan_target_mask[:, self.output_dims[target_name]]

            if len(target[avail_mask]) > 0:
                loss_per_target[target_name] = criterion_targets[target_name](
                    target_prediction[avail_mask, j], target[avail_mask].flatten()
                ).item()
                if self.target_names_and_types[target_name] == "binary":
                    # Apply sigmoid activation for binary classification
                    y_pred_class = (
                        torch.sigmoid(target_prediction[:, j]) >= 0.5
                    ).float()
                    targets[target_name]["prediction"] = y_pred_class[
                        avail_mask
                    ].numpy()
                    targets[target_name]["ground_truth"] = target[avail_mask].numpy()
                    targets[target_name]["scores"] = (
                        target_prediction[avail_mask, j].detach().numpy()
                    )

        return loss_per_target, targets
