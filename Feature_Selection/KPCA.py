#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import squareform
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt

# Perform nonlinear dimensionality reduction using Kernel PCA with an RBF (Gaussian) kernel
# 'gamma' controls the spread of the RBF kernel — larger gamma = more localized features
kpca = KernelPCA(kernel='rbf', gamma=15, fit_inverse_transform=True)
kpca.fit(X)  # Fit KPCA model on the feature matrix X (e.g., correlation-based features)

# Define the number of brain regions (AAL atlas has 116 regions)
N = 116

# Get the projection of samples onto the first 3 Kernel PCs
pcs_kpca = kpca.transform(X)[:, :3]  # Extract the first 3 nonlinear components

# Define the number of strongest connections to visualize
top_k = 50

# Create figure with 3 polar subplots for PC1, PC2, PC3
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6),
    subplot_kw=dict(polar=True),
    constrained_layout=True
)

# Compute global color scale limits (vmin, vmax) for consistent visualization
all_vals = []
for pc in pcs_kpca.T:
    # Construct a symmetric matrix by taking the outer product of the component vector
    pc_matrix = np.outer(pc, pc)
    pc_matrix = (pc_matrix + pc_matrix.T) / 2
    flat = pc_matrix[np.triu_indices(N, k=1)]  # Extract upper triangle
    all_vals.extend(flat)
vmax = np.max(np.abs(all_vals))
vmin = -vmax

# Visualize each of the first 3 Kernel PCs
for i, (pc, ax) in enumerate(zip(pcs_kpca.T, axes)):
    # Build the symmetric connectivity matrix for each component
    pc_matrix = np.outer(pc, pc)
    pc_matrix = (pc_matrix + pc_matrix.T) / 2

    # Keep only the top_k strongest connections
    flat = pc_matrix[np.triu_indices(N, k=1)]
    threshold = np.sort(np.abs(flat))[-top_k]  # Find cutoff value
    mask = np.abs(pc_matrix) < threshold
    pc_matrix_masked = pc_matrix.copy()
    pc_matrix_masked[mask] = 0  # Zero out weak connections

    # Plot the connectivity circle for this KPCA component
    plot_connectivity_circle(
        squareform(pc_matrix_masked[np.triu_indices(N, k=1)]),
        node_names=labels,       # AAL region names
        title=f'PC{i+1}',        # Component title
        n_lines=top_k,           # Number of connections drawn
        colormap='coolwarm',     # Red–blue color map
        ax=ax,
        show=False,
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
        textcolor='black',       # Black node labels
        facecolor='white'        # White plot background
    )

# Add a shared colorbar for all three subplots
cax = fig.add_axes([1, 0, 0.01, 0.2])  # Position: [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('', rotation=270, labelpad=15, color='black')
cbar.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

plt.show()

# Refit KPCA and compute explained variance ratio approximation
# Kernel PCA does not directly provide explained variance like linear PCA,
# but eigenvalues can be normalized to approximate variance contribution.
kpca = KernelPCA(kernel='rbf', gamma=15, fit_inverse_transform=True)
kpca.fit(X)
eigenvalues = kpca.eigenvalues_

# Normalize eigenvalues to get proportion of variance explained by each component
explained_variance_ratio_kpca = eigenvalues / np.sum(eigenvalues)

# Find number of components needed to explain at least 90% of the variance
n_components_kpca = np.argmax(np.cumsum(explained_variance_ratio_kpca) >= 0.99) + 1
print(f"90% variance contribution: {n_components_kpca}")

# Re-run KPCA with optimal number of components and extract transformed features
kpca = KernelPCA(n_components=n_components_kpca, kernel='rbf', gamma=15)
kpca_features = kpca.fit_transform(X)

# Display the shape of the reduced feature matrix (samples × selected components)
kpca_features.shape
