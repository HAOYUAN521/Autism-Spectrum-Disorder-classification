#!/usr/bin/env python
# coding: utf-8


import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping


class SimpleNetwork(pl.LightningModule):
    """
    Fully connected feedforward network for binary classification.

    Parameters:
        input_shape (int): Number of input features
        output_shape (int): Number of output neurons (1 for binary)
        hidden_sizes (list of int): Number of neurons in each hidden layer
        dropout_prob (float): Dropout probability for regularization
    """
    def __init__(self, input_shape, output_shape, hidden_sizes, dropout_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.flatten_layer = torch.nn.Flatten()  # Flatten input to 1D
        self.validation_step_outputs = []  # Buffer for per-batch validation results

        # Build sequential hidden layers with BatchNorm, GELU activation, and Dropout
        layers = []
        in_features = int(np.prod(input_shape))
        for hidden_size in hidden_sizes:
            layers += [
                torch.nn.Linear(in_features, hidden_size),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout_prob)
            ]
            in_features = hidden_size  # Update input for next layer

        self.hidden_layers = torch.nn.Sequential(*layers)
        self.output_layer = torch.nn.Linear(in_features, output_shape)

        # Loss function and metrics
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary classification loss
        self.accuracy = torchmetrics.classification.Accuracy(task='binary', num_classes=2)
        self.recall = BinaryRecall()  # Tracks recall
        self.specificity = BinarySpecificity()  # Tracks specificity

    def forward(self, x):
        """Forward pass"""
        x = self.flatten_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def predict(self, x):
        """Compute softmax probabilities for inference"""
        y = self.forward(x)
        y = torch.softmax(y, -1)
        return y

    def configure_optimizers(self):
        """Define optimizer"""
        return torch.optim.SGD(self.parameters(), lr=0.001)

    def training_step(self, train_batch, batch_idx):
        """Single training step"""
        x, y = train_batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)

        # Compute additional metrics
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        recall = self.recall(preds, y.int())
        specificity = self.specificity(preds, y.int())

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        x, y = batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(torch.sigmoid(y_pred.squeeze(1)), y.squeeze(1).long())

        # Compute additional metrics
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        recall = self.recall(preds, y.int())
        specificity = self.specificity(preds, y.int())

        self.log('val_acc', acc, on_epoch=True)
        self.validation_step_outputs.append({'val_loss': loss, 'val_acc': acc})
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        """Aggregate validation metrics over epoch"""
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        """Single test step"""
        x, y = test_batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)

        # Compute additional metrics
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        recall = self.recall(preds, y.int())
        specificity = self.specificity(preds, y.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return loss

# 5-Fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=14)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold + 1} ===")

    # Create data module for current fold
    data_module = FmriDataModule(train_indices=train_idx, val_indices=val_idx)
    data_module.setup(stage='fit')

    # Initialize network
    model = SimpleNetwork(
        input_shape=data_module.input_shape[0],
        output_shape=data_module.output_shape[0],
        hidden_sizes=[52, 20, 10],
        dropout_prob=0.1
    )

    # Trainer with early stopping
    logger = pl.loggers.CSVLogger("logs", name=f"simple_fold_{fold}")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=True)
    trainer = Trainer(
        max_epochs=200,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=0,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50), early_stopping]
    )

    # Train & validate
    trainer.fit(model, data_module)
    val_result = trainer.validate(model, data_module)

    # Store results
    fold_results.append({
        'fold': fold + 1,
        'val_loss': val_result[0]['val_loss'],
        'val_acc': val_result[0]['val_acc']
    })

# Summary of Cross-validation 
print("\n=== Cross-validation results ===")
for result in fold_results:
    print(f"Fold {result['fold']}: Val Loss = {result['val_loss']:.6f}, Val Acc = {result['val_acc']:.6f}")

# Compute average metrics
avg_loss = np.mean([r['val_loss'] for r in fold_results])
avg_acc = np.mean([r['val_acc'] for r in fold_results])

print(f"\nAverage across folds: Val Loss = {avg_loss:.6f}, Val Acc = {avg_acc:.6f}")

# Plot Validation Accuracy 
plt.figure(figsize=(5, 5))
for fold in range(5):
    try:
        log_base = f"logs/simple_fold_{fold}"
        versions = [d for d in os.listdir(log_base) if d.startswith('version_')]
        latest_version = sorted(versions)[-1] if versions else 'version_0'
        log_path = f"{log_base}/{latest_version}/metrics.csv"
        metrics = pd.read_csv(log_path)

        val_acc = metrics['val_acc'].dropna().reset_index(drop=True)
        plt.plot(val_acc, label=f'Fold {fold + 1}', linewidth=2)
    except Exception as e:
        print(f"Error loading fold {fold} data: {str(e)}")

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('Validation Accuracy Across 5 Folds', fontsize=14, pad=20)
plt.legend(fontsize=10, framealpha=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Skfolder.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot Validation Loss 
plt.figure(figsize=(5, 5))
for fold in range(5):
    try:
        log_base = f"logs/simple_fold_{fold}"
        versions = [d for d in os.listdir(log_base) if d.startswith('version_')]
        latest_version = sorted(versions)[-1] if versions else 'version_0'
        log_path = f"{log_base}/{latest_version}/metrics.csv"
        metrics = pd.read_csv(log_path)

        val_loss = metrics['val_loss'].dropna().reset_index(drop=True)
        plt.plot(val_loss, label=f'Fold {fold + 1}', linewidth=2)
    except Exception as e:
        print(f"Error loading fold {fold} data: {str(e)}")

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Validation Loss Across 5 Folds', fontsize=14, pad=20)
plt.legend(fontsize=10, framealpha=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Skfolder_loss.png', dpi=300, bbox_inches='tight')
plt.show()
