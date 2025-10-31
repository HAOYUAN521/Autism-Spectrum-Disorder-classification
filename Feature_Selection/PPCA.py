#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import squareform
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt

# Number of AAL brain regions
N = 116

# Number of strongest connections to visualize
top_k = 50

# ===============================
# Definition of the PPCA class
# ===============================
class PPCA:
    def __init__(self, n_components=None):
        # n_components: number of latent dimensions to retain
        self.n_components = n_components
        self.W = None             # Weight (loading) matrix
        self.sigma2 = None        # Variance of isotropic noise
        self.X_mean = None        # Mean of training data
        self.explained_variance_ratio_ = None  # Variance ratio for components

    def fit(self, X, max_iter=50, tol=1e-4):
        """
        Fit the PPCA model using the Expectation-Maximization (EM) algorithm.
        X: input data (samples × features)
        """
        n_samples, n_features = X.shape
        self.X_mean = np.mean(X, axis=0)
        X_centered = X - self.X_mean  # Center data

        # Initialize W using top PCA directions (via SVD)
        U, S, Vt = svd(X_centered, full_matrices=False)
        self.W = Vt[:self.n_components].T * np.sqrt((S[:self.n_components]**2 / n_samples).reshape(1, -1))
        self.sigma2 = np.mean((S[self.n_components:]**2) / n_samples) if self.n_components < len(S) else 1.0

        I_m = np.eye(self.n_components)

        # EM iteration
        for _ in range(max_iter):
            # ---------- E-step ----------
            # Compute posterior expectations of latent variables
            M = self.W.T @ self.W + self.sigma2 * I_m
            M_inv = np.linalg.inv(M)
            X_latent = X_centered @ self.W @ M_inv.T  # Expected latent variables (E[z|x])

            # ---------- M-step ----------
            # Update W and sigma^2 using expected sufficient statistics
            S = (X_centered.T @ X_centered) / n_samples
            W_new = S @ self.W @ np.linalg.inv(self.sigma2 * I_m + M_inv @ self.W.T @ S @ self.W)
            sigma2_new = (np.trace(S) - np.trace(W_new @ M_inv @ self.W.T @ S)) / n_features

            # ---------- Convergence check ----------
            if np.linalg.norm(W_new - self.W) < tol and abs(sigma2_new - self.sigma2) < tol:
                break

            self.W = W_new
            self.sigma2 = sigma2_new

        # Compute explained variance ratio after convergence
        self._calculate_explained_variance(X_centered)

    def _calculate_explained_variance(self, X_centered):
        """
        Compute explained variance ratio using SVD of centered data.
        """
        n_samples = X_centered.shape[0]
        _, s, _ = svd(X_centered, full_matrices=False)
        explained_variance = (s**2) / (n_samples - 1)
        total_variance = np.sum(explained_variance)
        self.explained_variance_ratio_ = explained_variance / total_variance

    def transform(self, X):
        """
        Project input data onto the PPCA latent space.
        """
        X_centered = X - self.X_mean
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.n_components)
        return np.linalg.solve(M.T, (X_centered @ self.W).T).T

    def get_components_with_variance_threshold(self, threshold=0.99):
        """
        Find the number of components needed to reach a specified cumulative variance ratio.
        """
        cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio_)
        return np.searchsorted(cumulative_variance_ratio, threshold) + 1


# ===============================
# Apply PPCA to dataset X
# ===============================

# Step 1: Fit PPCA to estimate variance structure and select components
ppca = PPCA(n_components=min(X.shape))
ppca.fit(X)

# Step 2: Determine how many components explain 99% of variance
n_components_99 = ppca.get_components_with_variance_threshold(threshold=0.99)
print(f"99% variance contribution: {n_components_99}")

# Step 3: Refit PPCA with optimal number of components
ppca = PPCA(n_components=n_components_99)
ppca.fit(X)
pcs_ppca = ppca.transform(X)  # Extract latent feature projections


# ===============================
# Visualization: Connectivity Circles
# ===============================

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True), constrained_layout=True)

# Compute global color scaling (vmin/vmax)
all_vals = []
for pc in pcs_ppca.T:
    mat = np.outer(pc, pc)
    mat = (mat + mat.T) / 2
    all_vals.extend(mat[np.triu_indices(N, k=1)])
vmax = np.max(np.abs(all_vals))
vmin = -vmax

# Plot first three PPCA components as circular brain networks
for i, (pc, ax) in enumerate(zip(pcs_ppca.T, axes)):
    mat = np.outer(pc, pc)
    mat = (mat + mat.T) / 2
    flat = mat[np.triu_indices(N, k=1)]
    threshold = np.sort(np.abs(flat))[-top_k]
    mask = np.abs(mat) < threshold
    mat_masked = mat.copy()
    mat_masked[mask] = 0

    # Draw connectivity circle
    plot_connectivity_circle(
        squareform(mat_masked[np.triu_indices(N, k=1)]),
        node_names=labels,
        title=f'PC{i+1}',
        n_lines=top_k,
        colormap='coolwarm',
        ax=ax,
        show=False,
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
        textcolor='black',
        facecolor='white'
    )

# Add shared colorbar for all subplots
cax = fig.add_axes([1, 0, 0.01, 0.2])
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

plt.show()
