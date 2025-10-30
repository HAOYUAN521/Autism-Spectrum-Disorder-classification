#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt

# Define the number of brain regions (AAL atlas has 116 regions)
N = 116

# Define the AAL116 region labels (left/right hemisphere and subcortical structures)
labels = [
    'Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R',
    'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R',
    'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R',
    'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
    'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R',
    'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
    'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
    'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R',
    'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R',
    'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
    'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R',
    'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
    'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R',
    'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R',
    'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
    'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
    'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
    'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R',
    'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R',
    'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R',
    'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R',
    'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',
    'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebellum_Crus1_L', 'Cerebellum_Crus1_R',
    'Cerebellum_Crus2_L', 'Cerebellum_Crus2_R', 'Cerebellum_3_L', 'Cerebellum_3_R',
    'Cerebellum_4_5_L', 'Cerebellum_4_5_R', 'Cerebellum_6_L', 'Cerebellum_6_R',
    'Cerebellum_7b_L', 'Cerebellum_7b_R', 'Cerebellum_8_L', 'Cerebellum_8_R',
    'Cerebellum_9_L', 'Cerebellum_9_R', 'Cerebellum_10_L', 'Cerebellum_10_R',
    'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6',
    'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10'
]

# Perform PCA on the feature matrix X
# X should contain all subjects' upper-triangle correlation vectors (from previous code)
pca = PCA()
pca.fit(X)

# Extract the first 3 principal components (PC1, PC2, PC3)
pcs = pca.components_[0:3]

# Define the number of strongest connections to display
top_k = 50

# Create a figure with 3 polar subplots for PC1, PC2, and PC3
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6),  # Width = 18 inches, height = 6 inches
    subplot_kw=dict(polar=True),
    constrained_layout=True  # Automatically adjust layout to prevent overlap
)

# Calculate global color scaling (vmin/vmax) for consistent color range across plots
all_vals = []
for pc in pcs:
    pc_matrix = squareform(pc)         # Convert the flattened PC vector back to a symmetric matrix
    pc_matrix = pc_matrix + pc_matrix.T  # Make it fully symmetric
    flat = pc_matrix[np.triu_indices(N, k=1)]  # Extract upper triangle values
    all_vals.extend(flat)
vmax = np.max(np.abs(all_vals))
vmin = -vmax

# Loop through each principal component and visualize its strongest connections
for i, (pc, ax) in enumerate(zip(pcs, axes)):
    pc_matrix = squareform(pc)
    pc_matrix = pc_matrix + pc_matrix.T

    # Identify the top_k strongest connections (by absolute weight)
    flat = pc_matrix[np.triu_indices(N, k=1)]
    threshold = np.sort(np.abs(flat))[-top_k]
    mask = np.abs(pc_matrix) < threshold
    pc_matrix_masked = pc_matrix.copy()
    pc_matrix_masked[mask] = 0  # Zero out weaker connections

    # Plot the connectivity circle for each principal component
    plot_connectivity_circle(
        pc_matrix_masked,
        node_names=labels,
        title=f'PC{i+1}',
        n_lines=top_k,
        colormap='coolwarm',
        ax=ax,
        show=False,
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
        textcolor='black',   # Node labels in black
        facecolor='white'    # White background
    )

# Add a shared colorbar for all subplots
cax = fig.add_axes([1, 0, 0.01, 0.2])  # Position of colorbar [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('', rotation=270, labelpad=15, color='black')
cbar.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

# Display the final figure
plt.show()

# Compute cumulative explained variance ratio to find how many PCs explain 99% of the variance
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
print(f"99% variance contribution: {n_components_99}")

# Refit PCA with the optimal number of components (99% variance)
pca = PCA(n_components=n_components_99)
pca_features = pca.fit_transform(X)

# Show the shape of the reduced feature matrix (subjects × components)
pca_features.shape
