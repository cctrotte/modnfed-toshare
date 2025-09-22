# example training script for modn with very simple data (no FL)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from modules_modn import ModularModel
import seaborn as sns


if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Step 1: Generate synthetic data
    n_samples = 100
    n_features = 3
    feature_names = ["feature1", "feature2", "feature3"]
    X = torch.randn(n_samples, n_features)
    y_1 = ((X.sum(dim=1) + torch.randn(n_samples) * 0.1).view(-1, 1) > 0).float()
    weights = torch.randn(X.shape[1])
    y_2 = ((X @ weights + torch.randn(n_samples) * 0.1).view(-1, 1) > 0).float()
    y = torch.cat((y_1, y_2), dim=1)

    # Hyperparameters
    state_dim = 2
    learning_rate = 0.01
    num_epochs = 200

    # Instantiate the model, define the loss function and the optimizer
    model = ModularModel(
        feature_names=feature_names,
        state_dim=state_dim,
        predict_all=True,
        input_dims={f: [i] for i, f in enumerate(feature_names)},
        target_names_and_types={"target_1": "binary", "target_2": "binary"},
        output_dims={"target_1": 0, "target_2": 1},
    )



    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss() #nn.MSELoss()

    # Step 4: Training the model with debugging loss
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed_features, predicted_targets,_ = model(X)
        if model.predict_all:
            # adapt size of target since the model predicts each time a new feature is encoded
            y_stacked = y.repeat(len(feature_names)+1, 1, 1)
        else:
            y_stacked = y
        # Compute standard losses
        # loss_features = criterion(reconstructed_features, X)

        if predicted_targets.shape != y_stacked.shape:
            raise ValueError(
                f"Shape mismatch: predicted shape is {predicted_targets.shape}, "
                f"but target shape is {y_stacked.shape}"
            )

        #loss_target = criterion(predicted_target, y_stacked)
        loss_targets = 0
        for j, target_name in enumerate(["target_1", "target_2"]):
            target = y_stacked[..., model.output_dims[target_name]]
            prediction = predicted_targets[..., j]
            loss_target = criterion(
                prediction[..., :], target[..., :]
            )
            loss_targets += loss_target
        # loss = loss_features + loss_target

        # Backward pass and optimization
        loss_targets.backward()
        optimizer.step()

        # Compute debugging loss after encoding all features
        if (epoch + 1) % 10 == 0:

            # Print the loss after encoding all features using the debugging method
            losses, _ = model.compute_loss_after_encoding(
                X,
                y,
                feature_names,
                ["target_1", "target_2"],
                torch.isnan(X),
                torch.isnan(y),
                criterion_targets={"target_1": criterion, "target_2": criterion},
            )  # compute_loss_after_encoding(X, y, feature_names, nan_mask)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Target Training Loss: {loss_target.item():.4f}, Training loss after encoding all features: {losses[-1][2]:.4f}"
            )
    
    # plot heatmap of predicted probabilities
    i_patient = 1
    gt = y[i_patient]

    _, pred, feat_order = model(X[i_patient,:].reshape(1,-1))
    pred = torch.sigmoid(pred).detach().numpy()
    import matplotlib.pyplot as plt

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pred.squeeze(1).T, annot=True, fmt=".2f", cmap="viridis", yticklabels=[f"Target 1 (ground truth: {gt[0]})", f"Target 2 (ground truth: {gt[1]})"], xticklabels=[f"feature {j}" for j in feat_order])
    plt.title("Evolving Predictions as Features are Encoded")
    plt.ylabel("Predicted Target")
    plt.xlabel("Feature Encoding Order")
    plt.savefig("heatmap.png", dpi=300)
    plt.show()

    print("End")

