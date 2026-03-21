"""Transfer-learning model utilities for Intel Image Classification."""

import copy
import time

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=6, pretrained=True, freeze_backbone=True):
    """Build a ResNet-18 with a replaced fully-connected head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        Load ImageNet-pretrained weights.
    freeze_backbone : bool
        If True, freeze all layers except the final classifier.

    Returns
    -------
    model : nn.Module
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_layers(model, num_layers=0):
    """Unfreeze the last *num_layers* residual blocks for fine-tuning.

    Pass ``num_layers=0`` to unfreeze only the classifier (default state).
    Use a higher number (1-4) to progressively unfreeze deeper layers.
    """
    children = list(model.children())
    all_blocks = [c for c in children if isinstance(c, nn.Sequential)]

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    if num_layers > 0:
        blocks_to_unfreeze = all_blocks[-num_layers:]
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True


def get_trainable_params(model):
    """Return the number and list of trainable parameter groups."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainable)
    return total, trainable


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run a single training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and return loss and accuracy.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, device="cpu", num_epochs=10, verbose=True):
    """Full training loop with optional LR scheduler and early-stopping checkpoint.

    Returns
    -------
    model : nn.Module
        The model with the best validation accuracy weights restored.
    history : dict
        Keys: train_loss, val_loss, train_acc, val_acc, lr.
    """
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "lr": [],
    }
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    start = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        if verbose:
            elapsed = time.time() - start
            print(
                f"Epoch [{epoch+1:>2}/{num_epochs}]  "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
                f"LR: {current_lr:.2e}  ({elapsed:.0f}s)"
            )

    model.load_state_dict(best_weights)
    if verbose:
        print(f"\nBest val accuracy: {best_acc:.4f}")
    return model, history


@torch.no_grad()
def evaluate(model, loader, device, class_names=None):
    """Compute predictions and ground-truth labels on a dataset.

    Returns
    -------
    all_preds : list[int]
    all_labels : list[int]
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


@torch.no_grad()
def predict_single(model, image_tensor, device, class_names=None):
    """Predict the class of a single image tensor.

    Parameters
    ----------
    image_tensor : torch.Tensor of shape (C, H, W)

    Returns
    -------
    predicted_class : str or int
    probabilities : torch.Tensor of shape (num_classes,)
    """
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1).squeeze()
    idx = probs.argmax().item()

    if class_names is not None:
        return class_names[idx], probs
    return idx, probs


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, num_classes=6, device="cpu"):
    model = create_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
