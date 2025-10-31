#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Combine PCA, KPCA, and PPCA features into one feature matrix
X_combined = np.hstack([pca_features, kpca_features, ppca_features])

# Convert data to PyTorch tensors
X = np.array(X_combined)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=4)
X, y = smote.fit_resample(X, y)


class FmriDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for handling fMRI data."""

    def __init__(self, batch_size=21, num_workers=4, train_indices=None, val_indices=None, **kwargs):
        """
        Initialize the DataModule.

        Args:
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of parallel workers for data loading.
            train_indices (list): Indices of training samples.
            val_indices (list): Indices of validation samples.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_shape = None
        self.output_shape = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.train_indices = train_indices
        self.val_indices = val_indices

    def setup(self, stage: str):
        """Prepare the training, validation, and test datasets."""
        if (stage == 'fit' or stage == 'validate') and not (self.data_train and self.data_val):
            # Use provided indices to split the dataset
            x_train, y_train = X[self.train_indices], y[self.train_indices]
            x_val, y_val = X[self.val_indices], y[self.val_indices]
            
            # Record data shapes for later reference
            self.input_shape = (x_train.shape[1],)
            self.output_shape = (1,)
            
            # Convert into lists of (input, label) pairs
            self.data_train = list(zip(x_train, y_train))
            self.data_val = list(zip(x_val, y_val))
            self.data_test = self.data_val  # Optional: use validation data as test set

    def train_dataloader(self):
        """Return the DataLoader for training."""
        return torch.utils.data.DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        """Return the DataLoader for validation."""
        return torch.utils.data.DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        """Return the DataLoader for testing."""
        return torch.utils.data.DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        """Return the DataLoader for predictions or inference."""
        return torch.utils.data.DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )