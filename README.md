# Autism Spectrum Disorder classification: A study of using fused features based on deep learning models
This study proposes a novel feature fusion method to enhance the ASD classification performance by integrating the principal component features extracted by three dimensionality reduction techniques. The study evaluated the classification effects of three baseline deep learning models. The highest classification accuracy reached 80.2%. This proves that the use of fused features is meaningful.

## Description
Autism Spectrum Disorder (ASD) is a complex neurodevelopmental disease that mainly affects patients' social interaction, communication ability, and behavior patterns. Early and accurate diagnosis is crucial for intervention treatment, but existing diagnostic methods rely mainly on subjective evaluation. Resting-state functional magnetic resonance imaging (rs-fMRI) provides a powerful tool for objective classification of ASD by capturing the functional connectivity patterns of the brain. However, problems such as high-dimensional data, limited sample size and class imbalance restrict the improvement of model performance.

## Getting Started

### Data Load
- **Dataset**: ABIDE I (Autism Brain Imaging Data Exchange I)  
- **Provider**: Preprocessed Connectomes Project (PCP) [1] (http://preprocessed-connectomes-project.org/abide/)  
- **Download Method**: Official script `download_abide_preproc.py` 
- **Command Used**:
  ```bash
  python download_abide_preproc.py \
      -d func_preproc \
      -p dparsf \
      -s filt_global \
      -o your_output_dir
  ```
### Libraries Used
Libraries used: PyTorch, TorchVision, Lightning, TorchMetrics, NiBabel, Nilearn, SciPy, NumPy, Pandas, and Scikit-learn.

How to install the libraries used:
```bash
pip install -r requirements.txt
```


### Environment and Hardware

Python version used: 3.12.8

This project was run on the following GPU hardware:

- **GPU Model**: NVIDIA GeForce RTX 2080 Ti
- **Driver Version**: 570.124.06
- **CUDA Version**: 12.8
- **Total GPU Memory**: 11,264 MiB


















### Reference
[1] Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden.
