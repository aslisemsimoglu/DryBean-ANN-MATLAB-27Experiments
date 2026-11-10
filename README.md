# Dry Bean Classification with Artificial Neural Networks in MATLAB

## Overview

This repository documents a controlled comparison of three feedforward artificial neural network (ANN) architectures for seven-class dry bean classification using MATLAB. The experimental design comprises **27 training runs**: three architectures × three nominal training-pool/test splits × three random seeds.

## Dataset

The Dry Bean Dataset contains **13,611 observations**, **16 numerical features**, and **7 bean classes**. The source dataset and a saved MATLAB split artifact are located in `01_Veri/`.

**Dataset reference:** Koklu, M., & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507. https://doi.org/10.1016/j.compag.2020.105507

Original data: [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).

## Network architectures

| Model | Architecture | Hidden activation | Output |
|---|---|---|---|
| ANN-1 | 16 → 10 → 7 | ReLU | Softmax |
| ANN-2 | 16 → 20 → 10 → 7 | ReLU | Softmax |
| ANN-3 | 16 → 32 → 16 → 8 → 7 | ReLU | Softmax |

## Experimental protocol

- Environment: MATLAB R2026b; NVIDIA RTX 4050 Laptop GPU for 26 runs and CPU for one initial run.
- Optimizer: Adam; initial learning rate: 0.001.
- Maximum epochs: 50; mini-batch size: 128.
- Loss: cross-entropy; evaluation metrics include accuracy and macro-F1.
- Random seeds for independent training repeats: `101`, `202`, `303`.
- Normalization: feature-wise z-score statistics fitted to the training subset only.
- Validation: 10% of the nominal training pool, held out separately from testing.

| Nominal training pool / test | Actual train | Validation | Test |
|---|---:|---:|---:|
| 60/40 | 7,351 | 816 | 5,444 |
| 70/30 | 8,576 | 952 | 4,083 |
| 80/20 | 9,801 | 1,088 | 2,722 |

The nominal ratios refer to the training **pool** (train + validation) versus the test set, not the final train-only fraction.

## Results

Mean test accuracy over three repeats (%, rounded):

| Model | 60/40 | 70/30 | 80/20 |
|---|---:|---:|---:|
| ANN-1 | 92.481 | 92.024 | 91.465 |
| ANN-2 | 92.726 | 92.571 | 92.003 |
| ANN-3 | **92.946** | 92.465 | **92.138** |

The best individual recorded test accuracy was **93.0749%** (ANN-3, 60/40, seed 303). The full per-run results and group statistics are in `06_Sonuc_Tablolari/`. Comparisons across different split ratios involve different test sets and should not be interpreted as a controlled training-size effect in isolation.

## Repository structure

```text
01_Veri/                       Dataset and saved data partitions
02_Kodlar/                     Main MATLAB training/evaluation script
03_Modeller/                   27 saved model artifacts (.mat)
04_Egitim_Grafikleri/          27 accuracy + 27 loss plots
05_Karisiklik_Matrisleri/      27 confusion-matrix plots
06_Sonuc_Tablolari/            Per-run, per-class, and summary metrics
08_Karsilastirma_Grafikleri/   Performance and convergence comparisons
```

## Reproducing and inspecting the work

1. Install a compatible MATLAB version with the required Deep Learning Toolbox functionality and obtain the Dry Bean Dataset.
2. Open the repository root as the MATLAB working directory.
3. Inspect `01_Veri/DryBean_VeriBolmeleri.mat` and `02_Kodlar/YSA_03_Ana_Deneyler_GPU.m` before running. The original script was developed with a specific local folder layout and may require path or execution-environment adjustments.
4. To review the published results without retraining, open `06_Sonuc_Tablolari/Tum_Deney_Sonuclari.csv`, `Dokuz_Grup_Istatistik_Ozeti.csv`, and the PNG figures.

**Reproducibility note:** The saved partition file is included, but a standalone script that regenerates those partitions from raw data is not currently part of this publication package. The training script also includes resume-from-checkpoint behavior; inspect it before executing to avoid unexpected writes or training runs. Exact numerical reproducibility may vary with MATLAB release, device, and GPU libraries.

## Interpretation and limitations

- Each configuration was trained three times, allowing descriptive mean and sample standard deviation reporting; three repeats do not establish statistical significance.
- One run was trained on CPU and the remaining 26 on GPU, so runtime comparisons are not hardware-controlled.
- The results are specific to this dataset, preprocessing, split protocol, and model configurations.
- A higher final validation loss than the minimum recorded validation loss is not by itself proof of overfitting.

## Attribution

Please cite the original Dry Bean Dataset publication above when using the dataset. The MATLAB experiment code and generated results are part of this course project; no software license is granted unless a separate license file is added.
