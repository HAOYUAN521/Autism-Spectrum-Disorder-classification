#!/usr/bin/env python
# coding: utf-8

import os
import nibabel as nib
import re
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets

# Set the directory containing preprocessed fMRI images
fmri_dir = "/projects/hw4m/Fmri/fmri_image/Outputs/dparsf/filt_global/func_preproc"
# Set the output directory for correlation matrices
output_dir = "/projects/hw4m/Fmri/correlation_matrices"

# Load the AAL atlas for brain region parcellation
aal_atlas = datasets.fetch_atlas_aal()
template_path = aal_atlas['maps']   # Path to the atlas image
labels = aal_atlas['labels']        # Region labels

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Define a masker to extract mean time series from each atlas region
masker = NiftiLabelsMasker(labels_img=template_path, standardize=True)

# Loop through each subject’s fMRI file
for fmri_file in os.listdir(fmri_dir):
    if fmri_file.endswith(".nii.gz"):  # Only process NIfTI files
        fmri_path = os.path.join(fmri_dir, fmri_file)
        try:
            # Load the fMRI image
            fmri_img = nib.load(fmri_path)

            # Extract regional mean time series using the AAL atlas
            time_series = masker.fit_transform(fmri_img)

            # Compute the correlation matrix between all regions
            correlation_matrix = np.corrcoef(time_series.T)

            # Set the diagonal to zero to remove self-correlations
            np.fill_diagonal(correlation_matrix, 0)

            # Get the subject ID from the filename
            subject_id = os.path.splitext(fmri_file)[0]

            # Save the correlation matrix as a CSV file
            output_path = os.path.join(output_dir, f"{subject_id}_correlation.csv")
            pd.DataFrame(correlation_matrix).to_csv(output_path, index=False)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"save {fmri_file} wrong: {e}")

# Collect all correlation matrix file paths
correlation_dir = '/projects/hw4m/Fmri/correlation_matrices/'
file_paths = [os.path.join(correlation_dir, f) for f in os.listdir(correlation_dir) if f.endswith('.csv')]

# Load the ABIDE phenotype data
info = pd.read_csv("/projects/hw4m/Fmri/Phenotypic_V1_0b_preprocessed1.csv")

# Keep only subject ID and diagnosis group
label = info[['SUB_ID', 'DX_GROUP']]
label.loc[:, 'SUB_ID'] = label['SUB_ID'].astype(str)

# Regular expression to extract subject ID from filename
pattern = re.compile(r'(?<=_00)\d+')

# Initialize containers for features and labels
all_vectors_with_labels = []
all_id = []

# Iterate through each correlation matrix file
for file_path in file_paths:
    filename = os.path.basename(file_path)

    # Extract subject ID (7-digit number)
    ids_in_file = re.findall(r'\d{7}', filename)
    all_id.extend([id_[2:] for id_ in ids_in_file])

    if ids_in_file:
        sub_id = ids_in_file[0][2:]
    else:
        print(f"Warning: No valid SUB_ID found in the filename {filename}. Skipping this file.")
        continue

    # Match the subject ID with the diagnosis label
    label_row = label[label['SUB_ID'] == sub_id]

    if not label_row.empty:
        sub_in_smp = label_row['DX_GROUP'].values[0]  # Diagnosis group (1 = control, 2 = ASD)
        corr_matrix = pd.read_csv(file_path, header=None, skiprows=1).values

        # Extract the upper triangle (since the matrix is symmetric)
        upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
        flattened_vector = corr_matrix[upper_triangle_indices]

        # Append the feature vector with its label
        all_vectors_with_labels.append(np.append(flattened_vector, sub_in_smp))
    else:
        print(f"Warning: No label found for SUB_ID {sub_id}. Skipping this file.")

# Convert to NumPy array
all_vectors_with_labels = np.array(all_vectors_with_labels)

# Inspect a specific vector for data integrity
print(f"Vector at index 384:\n{all_vectors_with_labels[384]}")

# Check for missing or infinite values
has_nan = np.isnan(all_vectors_with_labels[384]).any()
has_inf = np.isinf(all_vectors_with_labels[384]).any()
print(f"Contains NaN: {has_nan}, Contains Inf: {has_inf}")

# Print lengths for consistency check
print(f"Length of vector at 384: {len(all_vectors_with_labels[384])}")
print(f"Length of first vector: {len(all_vectors_with_labels[0])}")

# Remove problematic sample (index 384)
all_vectors_with_labels = np.delete(all_vectors_with_labels, 384, axis=0)

print(f"All vectors with labels shape: {all_vectors_with_labels.shape}")

# Filter phenotype info for subjects that appear in correlation data
filtered_df = info[info['SUB_ID'].astype(str).isin(all_id)]

# Group by data collection site
grouped = filtered_df.groupby('SITE_ID')

# Count sex distribution per site
sex_counts = grouped['SEX'].value_counts().unstack(fill_value=0)
sex_counts.columns = [f'SEX_{col}_count' for col in sex_counts.columns]

# Compute mean and standard deviation of age per site
age_stats = grouped['AGE_AT_SCAN'].agg(['mean', 'std'])
age_stats.columns = ['AGE_AT_SCAN_mean', 'AGE_AT_SCAN_std']

# Count diagnosis groups per site
sub_in_smp_counts = grouped['DX_GROUP'].value_counts().unstack(fill_value=0)
sub_in_smp_counts.columns = [f'DX_GROUP_{col}_count' for col in sub_in_smp_counts.columns]

# Count combined SEX × DX_GROUP per site
sex_sub_counts = (
    filtered_df.groupby(['SITE_ID', 'DX_GROUP'])['SEX']
    .value_counts()
    .unstack(fill_value=0)
)

# Flatten multi-level columns
sex_sub_counts = sex_sub_counts.unstack(level=1, fill_value=0)
sex_sub_counts.columns = [
    f'SEX_{sex}_in_DX_GROUP_{sub_in_smp}_count'
    for sex, sub_in_smp in sex_sub_counts.columns
]

# Merge all site-level statistics
result = pd.concat([sex_counts, age_stats, sub_in_smp_counts, sex_sub_counts], axis=1).reset_index()

# Print site-level summary
print(result)

# Separate features and labels
X = all_vectors_with_labels[:, :-1]  # Correlation features
y = all_vectors_with_labels[:, -1]   # Diagnosis labels

# Check dimensions
print(X.shape)
print(y.shape)
