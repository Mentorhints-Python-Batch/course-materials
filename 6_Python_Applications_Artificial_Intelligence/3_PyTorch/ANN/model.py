"""ANN model definition, training, evaluation, and persistence utilities."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn


class CarPurchaseANN(nn.Module):
    """Feedforward neural network for car purchase amount regression.

    Parameters
    ----------
    input_dim : int
        Number of input features (default 5).
    hidden_dims : tuple[int, ...]
        Sizes of hidden layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(self, input_dim=5, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run a single training epoch.

    Returns
    -------
    avg_loss : float
    """
    model.train()
    running_loss = 0.0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        total += X_batch.size(0)

    return running_loss / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and return average loss.

    Returns
    -------
    avg_loss : float
    """
    model.eval()
    running_loss = 0.0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        running_loss += loss.item() * X_batch.size(0)
        total += X_batch.size(0)

    return running_loss / total


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, device="cpu", num_epochs=100, verbose=True):
    """Full training loop with optional LR scheduler and best-model checkpoint.

    Returns
    -------
    model : nn.Module
        Model with the best validation loss weights restored.
    history : dict
        Keys: train_loss, val_loss, lr.
    """
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    start = time.time()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start
            print(
                f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                f"Train Loss: {train_loss:.6f}  |  "
                f"Val Loss: {val_loss:.6f}  |  "
                f"LR: {current_lr:.2e}  ({elapsed:.0f}s)"
            )

    model.load_state_dict(best_weights)
    if verbose:
        print(f"\nBest val loss: {best_loss:.6f}")
    return model, history


@torch.no_grad()
def predict(model, loader, device):
    """Generate predictions for an entire DataLoader.

    Returns
    -------
    predictions : np.ndarray of shape (n,)
    targets     : np.ndarray of shape (n,)
    """
    model.eval()
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y_batch.numpy())

    return np.concatenate(all_preds).flatten(), np.concatenate(all_targets).flatten()


@torch.no_grad()
def predict_single(model, features_tensor, device):
    """Predict for a single sample.

    Parameters
    ----------
    features_tensor : torch.Tensor of shape (n_features,)

    Returns
    -------
    prediction : float  (scaled value)
    """
    model.eval()
    x = features_tensor.unsqueeze(0).to(device)
    return model(x).item()


def compute_metrics(y_true, y_pred):
    """Compute regression metrics in original scale.

    Returns
    -------
    dict with keys: mse, rmse, mae, r2
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def save_model(model, path):
    """Save model state dict to disk."""
    torch.save(model.state_dict(), path)


def load_model(path, input_dim=5, hidden_dims=(64, 32), dropout=0.2, device="cpu"):
    """Load a saved CarPurchaseANN from disk."""
    model = CarPurchaseANN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
