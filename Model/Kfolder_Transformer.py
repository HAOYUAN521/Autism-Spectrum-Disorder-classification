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


class Transformer(pl.LightningModule):
    """
    Transformer-based neural network model for binary classification.

    Parameters:
        input_dim (int): Number of input features (e.g., PCA/KPCA/PPCA combined)
        output_dim (int): Number of output classes (1 for binary classification)
        num_heads (int): Number of attention heads in the Transformer
        num_layers (int): Number of Transformer encoder layers
        dim_feedforward (int): Dimension of the feedforward layer
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=6, dim_feedforward=32, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.validation_step_outputs = []  # buffer for collecting validation metrics

        # Embedding layer: project input to latent space with normalization, activation, and dropout
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dim_feedforward),
            torch.nn.LayerNorm(dim_feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )

        # Transformer encoder layer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward*2,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_feedforward),
            torch.nn.Linear(dim_feedforward, output_dim),
            torch.nn.Dropout(dropout)
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()  # binary classification loss
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=2)  # accuracy metric

    def forward(self, x):
        """Forward pass: embedding -> Transformer -> classification head"""
        x = self.embedding(x)
        x = x.unsqueeze(1)  # add sequence dimension for Transformer
        x = self.transformer_encoder(x)
        x = x.squeeze(1)    # remove sequence dimension
        x = self.fc(x)
        return x

    def predict(self, x):
        """Compute softmax probabilities for inference"""
        y = self.forward(x)
        y = torch.softmax(y, -1)
        return y

    def configure_optimizers(self):
        """Use SGD optimizer with fixed learning rate"""
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer

    def predict_step(self, predict_batch, batch_idx):
        """Define prediction behavior for dataloader"""
        x, y = predict_batch
        y_pred = self.predict(x)
        return y_pred, y

    def training_step(self, train_batch, batch_idx):
        """Single training step: compute loss and accuracy"""
        x, y = train_batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step: compute loss and accuracy for the batch"""
        x, y = batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(torch.sigmoid(y_pred.squeeze(1)), y.squeeze(1).long())
        self.validation_step_outputs.append({'val_loss': loss, 'val_acc': acc})
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        """Average batch metrics at the end of validation epoch"""
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
        self.validation_step_outputs.clear()  # clear buffer

    def test_step(self, test_batch, batch_idx):
        """Compute test metrics similar to validation"""
        x, y = test_batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return loss


# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=4)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold + 1} ===")

    # Create data module using current fold's train/val indices
    data_module = FmriDataModule(train_indices=train_idx, val_indices=val_idx)
    data_module.setup(stage='fit')

    # Initialize model
    model = Transformer(input_dim=data_module.input_shape[0], output_dim=data_module.output_shape[0])

    # Logger and EarlyStopping callback
    logger = pl.loggers.CSVLogger("logs", name=f"transformer_fold_{fold}")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=True)

    trainer = Trainer(
        max_epochs=200,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=0,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50), early_stopping]
    )

    # Train and validate
    trainer.fit(model, data_module)
    val_result = trainer.validate(model, data_module)

    fold_results.append({
        'fold': fold + 1,
        'val_loss': val_result[0]['val_loss'],
        'val_acc': val_result[0]['val_acc'],
    })

# Summarize cross-validation results
print("\n=== Cross-validation results ===")
for result in fold_results:
    print(f"Fold {result['fold']}: Val Loss = {result['val_loss']:.6f}, Val Acc = {result['val_acc']:.6f}")

avg_loss = np.mean([r['val_loss'] for r in fold_results])
avg_acc = np.mean([r['val_acc'] for r in fold_results])
print(f"\nAverage across folds: Val Loss = {avg_loss:.6f}, Val Acc = {avg_acc:.6f}")

# Plot validation accuracy across folds
plt.figure(figsize=(5,5))
for fold in range(5):
    try:
        log_base = f"logs/transformer_fold_{fold}"
        versions = [d for d in os.listdir(log_base) if d.startswith('version_')]
        latest_version = sorted(versions)[-1] if versions else 'version_0'
        log_path = f"{log_base}/{latest_version}/metrics.csv"
        metrics = pd.read_csv(log_path)
        val_acc = metrics['val_acc'].dropna().reset_index(drop=True)
        plt.plot(val_acc, label=f'Fold {fold+1}', linewidth=2)
    except Exception as e:
        print(f"Error loading fold {fold} data: {str(e)}")

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('Validation Accuracy Across 10 Folds', fontsize=14, pad=20)
plt.legend(fontsize=10, framealpha=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Tkfolder.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot validation loss across folds
plt.figure(figsize=(5,5))
for fold in range(5):
    try:
        log_base = f"logs/transformer_fold_{fold}"
        versions = [d for d in os.listdir(log_base) if d.startswith('version_')]
        latest_version = sorted(versions)[-1] if versions else 'version_0'
        log_path = f"{log_base}/{latest_version}/metrics.csv"
        metrics = pd.read_csv(log_path)
        val_loss = metrics['val_loss'].dropna().reset_index(drop=True)
        plt.plot(val_loss, label=f'Fold {fold+1}', linewidth=2)
    except Exception as e:
        print(f"Error loading fold {fold} data: {str(e)}")

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Validation Loss Across 10 Folds', fontsize=14, pad=20)
plt.legend(fontsize=10, framealpha=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Tkfolder_loss.png', dpi=300, bbox_inches='tight')
plt.show()
