# README: TMCD --- Robust Covariance Estimation for Tensor-Valued Observations

This repo provides code/data for Master’s thesis on **TMCD** (Tensor Minimum Covariance Determinant), a robust procedure for 3D+ tensor data. Implementations are in **R** (CPU) and **Python/PyTorch** (GPU).

## Structure
- **code/** --- Core TMCD implementation (`tmcd_functions.R`, `tmcd_functions_gpu.py`).
- **comparison/** --- R notebooks comparing TMCD functions to classical MLE and synthetic-data generators.
- **real-world-examples/** --- TMCD applied to real data.
- **simulations/** --- TMCD tested on synthetic tensors drawn from a tensor-normal distribution.
- **LICENSE** --- License details.
- **README.md** --- This file.




