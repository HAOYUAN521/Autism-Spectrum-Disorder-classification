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


class ResidualLayer(pl.LightningModule):
    """
    Residual block consisting of Linear -> BatchNorm -> GELU -> Dropout
    with a skip connection (residual addition).
    
    Parameters:
        hidden_size (int): Number of neurons in the layer
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, hidden_size, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.residual = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.GELU()
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the residual layer:
        1. Apply linear -> batch norm -> activation -> dropout
        2. Add input x to output (skip connection)
        """
        y = self.residual(x)
        y = self.batch_norm(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = y + x
        return y

class ResidualNetwork(pl.LightningModule):
    """
    Residual feedforward network for binary classification.
    
    Parameters:
        input_shape (int): Dimension of input features
        output_shape (int): Number of output neurons (1 for binary classification)
        hidden_size (int): Number of neurons in hidden layers
        num_hidden_layers (int): Number of residual layers stacked
    """
    def __init__(self, input_shape, output_shape, hidden_size, num_hidden_layers, **kwargs):
        super().__init__(**kwargs)
        self.flatten_layer = torch.nn.Flatten()
        self.validation_step_outputs = []

        # Initial linear projection from input dimension to hidden size
        layers = [torch.nn.Linear(np.prod(input_shape), hidden_size)]
        # Add residual layers
        for _ in range(num_hidden_layers):
            layers.append(ResidualLayer(hidden_size))
        self.hidden_layers = torch.nn.Sequential(*layers)

        # Output layer: project hidden_size -> output_shape
        self.output_layer = torch.nn.Linear(hidden_size, output_shape)
        
        self.accuracy = torchmetrics.classification.Accuracy(task='binary', num_classes=2)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        """Forward pass through the residual network"""
        y = x
        y = self.flatten_layer(y)
        y = self.hidden_layers(y)
        y = self.output_layer(y)
        return y

    def predict(self, x):
        """Compute softmax probabilities for inference"""
        y = self.forward(x)
        y = torch.softmax(y, -1)
        return y

    def configure_optimizers(self):
        """Use SGD optimizer"""
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer

    def predict_step(self, predict_batch, batch_idx):
        """Single prediction step"""
        x, y = predict_batch
        y_pred = self.predict(x)
        return y_pred, y

    def training_step(self, train_batch, batch_idx):
        """Single training step"""
        x, y = train_batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)
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
        self.validation_step_outputs.append({'val_loss': loss, 'val_acc': acc})
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        """Average metrics across validation batches"""
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        """Compute test metrics"""
        x, y = test_batch
        y_pred = self.forward(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return loss

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=14)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold + 1} ===")
    data_module = FmriDataModule(train_indices=train_idx, val_indices=val_idx)
    data_module.setup(stage='fit')

    hidden_size = 40
    num_hidden_layers = 18
    model = ResidualNetwork(input_shape=data_module.input_shape[0],
                            output_shape=data_module.output_shape[0],
                            hidden_size=hidden_size,
                            num_hidden_layers=num_hidden_layers)

    logger = pl.loggers.CSVLogger("logs", name=f"residual_fold_{fold}")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=True)
    trainer = Trainer(max_epochs=200,
                      logger=logger,
                      enable_progress_bar=True,
                      log_every_n_steps=0,
                      callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50), early_stopping])

    trainer.fit(model, data_module)
    val_result = trainer.validate(model, data_module)

    fold_results.append({'fold': fold + 1,
                         'val_loss': val_result[0]['val_loss'],
                         'val_acc': val_result[0]['val_acc']})

# Print cross-validation results
print("\n=== Cross-validation results ===")
for result in fold_results:
    print(f"Fold {result['fold']}: Val Loss = {result['val_loss']:.6f}, Val Acc = {result['val_acc']:.6f}")

avg_loss = np.mean([r['val_loss'] for r in fold_results])
avg_acc = np.mean([r['val_acc'] for r in fold_results])
print(f"\nAverage across folds: Val Loss = {avg_loss:.6f}, Val Acc = {avg_acc:.6f}")

# Plot validation accuracy
plt.figure(figsize=(5, 5))
for fold in range(5):
    try:
        log_base = f"logs/residual_fold_{fold}"
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
plt.savefig('Rkfolder.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot validation loss
plt.figure(figsize=(5, 5))
for fold in range(5):
    try:
        log_base = f"logs/residual_fold_{fold}"
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
plt.savefig('Rkfolder_loss.png', dpi=300, bbox_inches='tight')
plt.show()