# MK-UNet: Codebase Exploration and Experimental Verification

## 1. Objective

In this study, the original implementation of [MK-UNet](https://arxiv.org/pdf/2509.18493) was explored, and reproducing the reported benchmarking result on `ClinicDB` and `ColonDB` was attempted. Alongside of trying to reproduce the reported result, the identical architecture was trained on a non-biomedical domain (`SugarBeets2016`, from the guidelines' dataset menu) data to study cross-domain behavior.

## 2. Environment and Setup Notes
- GPU - NVIDIA H100 (GCP)
- Python - 3.10
- PyTorch / CUDA - 2.14 / 13.0