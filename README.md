# Color-Space Diversity for Retinal Age Estimation

This repository contains the reproducible training and analysis workflow for a
study of color representations in retinal fundus age estimation. All models use
the same patient-disjoint data splits, ImageNet-pretrained ResNet-50
architecture, optimization protocol, and checkpoint-selection rule. Only the
input representation or RGB training seed changes.

The primary comparison asks whether an ensemble of different color spaces
provides more useful predictive diversity than an equal-size ensemble of RGB
models.

## Study Design

The canonical study contains 24 models:

- 20 representations trained with seed 42:
  - full `RGB`, `Lab`, `HSV`, and `YCrCb`
  - 12 single-channel ablations: `R`, `G`, `B`, `L`, `a`, `b`, `H`, `S`,
    `V`, `Y`, `Cr`, and `Cb`
  - three predefined Custom3 hybrid representations
  - one grayscale structural control
- four additional full-RGB models trained with seeds 43, 44, 45, and 46

The two primary ensembles are fixed and disjoint:

| Ensemble | Members |
| --- | --- |
| `Color4-42` | RGB seed42, Lab seed42, HSV seed42, YCrCb seed42 |
| `RGB4` | RGB seed43, RGB seed44, RGB seed45, RGB seed46 |

Both ensembles average four age predictions, have the same inference cost, and
share no model. The comparison concerns these two fixed ensembles; it is not an
estimate of performance over a population of arbitrary training seeds.

An additional **early fusion experiment (Ensemble²)** concatenates all four
color-space representations into a single 12-channel input tensor and trains
one ResNet-50 on the joint representation. This is a supplementary experiment
comparing early-fusion (input concatenation) against late-fusion (prediction
averaging) for multi-color-space age estimation.

## Dataset and Splits

The tracked manifests define patient-level train, validation, and test splits.
No patient, image ID, or resolved image path appears in more than one split.

| Split | Images | Patients | Age range | Mean age |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,902 | 3,775 | 5–97 | 57.42 |
| Validation | 1,493 | 809 | 8–92 | 57.77 |
| Test | 1,462 | 809 | 7–94 | 57.30 |
| Total | 9,857 | 5,393 | 5–97 | — |

Raw retinal images are not distributed in this repository. To reproduce the
study, provide the same source dataset with paths matching the manifests:

```text
<DATA_ROOT>/
└── images/
    ├── img00001.jpg
    ├── img00002.jpg
    └── ...
```

## Training Protocol

All 24 canonical models use:

- ImageNet-pretrained ResNet-50, fully fine-tuned
- 80 epochs and batch size 32
- AdamW with learning rate `1e-4` and weight decay `1e-4`
- gradient-norm clipping at `1.0`
- label-distribution-smoothing weights with Smooth L1 training loss
- ReduceLROnPlateau scheduling
- horizontal flipping with probability `0.5` and rotation within ±10°
- representation-specific normalization computed from the train split only
- best-checkpoint selection by validation MAE

The shared training template is
[`configs/resnet50_imagenet_local.yaml`](configs/resnet50_imagenet_local.yaml).
Representation contracts under `configs/` lock channel order, stored dtype,
numeric range, cache format, and tensor scaling.

The Ensemble² model follows the same protocol with one architectural difference:
its first convolutional layer accepts 12 channels (RGB + Lab + HSV + YCrCb).
Pretrained `conv1` weights are adapted by repeating them four times and scaling
by `1/4` to preserve the expected activation magnitude.

## Primary Results

MAE is reported in years.

| Split | Color4-42 MAE | Independent RGB4 MAE | RGB4 − Color4 |
| --- | ---: | ---: | ---: |
| Validation | **4.9658** | 5.1680 | **0.2022** |
| Test | **4.5983** | 4.7436 | **0.1453** |

On the test split, Color4 reduces MAE by 0.1453 years, or 3.06% relative to the
independent RGB4 control.

Uncertainty is estimated with a paired non-parametric image-level bootstrap.
Each of 100,000 repetitions samples 1,462 test images with replacement and
uses the same sampled indices for both ensembles.

- observed test improvement: **0.145294 years**
- 95% bootstrap interval: **[0.022534, 0.269069] years**
- proportion of bootstrap improvements greater than zero: **98.975%**
- bootstrap RNG seed: `20260720`

### Diversity analysis

Using the ambiguity decomposition
`ensemble MSE = mean member MSE − diversity`:

| Split | RGB4 diversity | Color4 diversity | Color4 / RGB4 |
| --- | ---: | ---: | ---: |
| Validation | 5.1772 | 8.8868 | **1.72×** |
| Test | 4.6006 | 8.8991 | **1.93×** |

The test diversity gain is 4.2985 with a 95% image-bootstrap interval of
`[3.6932, 4.9477]`; all bootstrap repetitions produce a positive diversity
gain. The result supports the interpretation that heterogeneous color
representations help through less-correlated member errors rather than through
a single non-RGB model outperforming RGB.

## Representation Results

### Full representations, seed 42

| Representation | Validation MAE | Test MAE |
| --- | ---: | ---: |
| RGB | 5.4090 | **4.9067** |
| YCrCb | **5.4046** | 5.1084 |
| Lab | 5.5419 | 5.2448 |
| HSV | 5.4661 | 5.2852 |
| Grayscale | 6.1552 | 5.9976 |

No non-RGB full representation outperforms RGB on the test split.

### Early fusion (Ensemble²)

A single ResNet-50 trained on all four color spaces concatenated into a
12-channel input tensor, using the same training protocol as the canonical
24 models (seed 42, ReduceLROnPlateau, 80 epochs).

| Model | Validation MAE | Test MAE |
| --- | ---: | ---: |
| Ensemble² (12-ch early fusion) | 5.6764 | 5.2039 |

Ensemble² does not outperform RGB (test MAE 4.9067) or Color4-42 (test MAE
4.5983). This is consistent with the observation that the four color spaces
are mathematically invertible transformations of each other, limiting the
additional information available to a single jointly-trained model. Late
fusion via independent prediction averaging (Color4-42) remains the more
effective combination strategy.

### Custom3 hybrids

The candidates were fixed before training. Each is a single three-channel
tensor evaluated by one ResNet-50, not an ensemble.

| Hybrid input | Validation MAE | Test MAE |
| --- | ---: | ---: |
| `[Lab-b, RGB-G, RGB-B]` | **5.3145** | **4.9753** |
| `[Lab-b, RGB-G, HSV-S]` | 5.3560 | 5.0363 |
| `[Lab-a, RGB-G, Lab-b]` | 5.3803 | 5.0752 |

The first candidate is selected by validation MAE. It does not outperform RGB
on test: RGB seed42 is better by 0.0686 years, and the paired image-bootstrap
interval for a Custom3 improvement is `[-0.2250, 0.0870]`.

### Single-channel ablations

Each selected channel is repeated three times to preserve the input shape
expected by the pretrained network.

| Representation | Validation MAE | Test MAE |
| --- | ---: | ---: |
| RGB-R | 7.0720 | 6.7047 |
| RGB-G | 6.1317 | 5.8333 |
| RGB-B | 6.5042 | 6.0592 |
| Lab-L | 6.4377 | 6.0677 |
| Lab-a | 6.7317 | 6.0522 |
| Lab-b | 6.5460 | 6.0295 |
| HSV-H | 6.5357 | 6.1345 |
| HSV-S | 6.3637 | 5.9151 |
| HSV-V | 7.1904 | 6.7435 |
| YCrCb-Y | **6.0825** | **5.6944** |
| YCrCb-Cr | 6.4902 | 6.1461 |
| YCrCb-Cb | 6.5453 | 5.9632 |

YCrCb-Y is the strongest single channel, followed by RGB-G. Every
single-channel model is worse than full RGB.

## Repository Structure

```text
retinal-color-transfer/
├── configs/                      Representation contracts and training template
├── data/
│   ├── manifests/                Patient-disjoint split definitions
│   └── normalization/
│       ├── <family>/             Per-representation train-split statistics
│       └── fusion/               12-channel fusion normalization statistics
├── notebooks/
│   └── train_all_models.ipynb    End-to-end Colab reproduction (incl. Ensemble²)
├── scripts/
│   ├── 01_prepare_training.py
│   ├── 02_evaluate_and_analyze.py
│   └── train_fusion12ch.py       Standalone Ensemble² training script
├── src/retinal_color_transfer/   Training and analysis package
├── pyproject.toml                Dependencies and package configuration
└── README.md
```

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Reproducing the Study

### 1. Prepare caches, normalization, and configs

```bash
PYTHONPATH=src python3 scripts/01_prepare_training.py \
  --data-root /path/to/retinal-data
```

The command:

1. builds the prepared RGB cache;
2. derives all full, single-channel, grayscale, and Custom3 caches;
3. reuses valid cache entries;
4. computes missing normalization statistics from the train split only;
5. validates all 9,857 entries for every representation; and
6. writes configs for the canonical 24 models under `configs/generated/`.

Generated artifacts follow the same family layout:

```text
caches/<family>/<representation>/
data/normalization/<family>/<representation>_train_stats.json
models/<family>/<representation>_seed<seed>/
```

### 2. Train all models

Use [`notebooks/train_all_models.ipynb`](notebooks/train_all_models.ipynb) in a
GPU-enabled Colab runtime. Set its repository URL, raw-data root, and persistent
Drive run root, then use **Run all**.

The notebook validates the raw data, runs preparation, verifies the 24 generated
configs, trains every canonical model, resumes an interrupted model from
`latest_checkpoint.pt`, skips completed models, and runs final analysis.

During training a model directory contains:

```text
best_checkpoint.pt
latest_checkpoint.pt
training_history.csv
```

After successful training, `latest_checkpoint.pt` is removed. After evaluation,
the finalized directory contains exactly:

```text
best_checkpoint.pt
predictions.csv
training_history.csv
```

### 3. Recreate predictions and analysis

```bash
PYTHONPATH=src python3 scripts/02_evaluate_and_analyze.py \
  --data-root /path/to/retinal-data
```

The workflow validates and reuses every existing `predictions.csv`. If one is
missing, it performs validation and test inference from `best_checkpoint.pt`.
It then writes model metrics, fixed Color4-vs-RGB4 bootstrap comparisons, and
ensemble-diversity results under `analysis/final_results/`.

## Reproducibility Scope

The repository tracks code, configs, split manifests, normalization statistics,
and the end-to-end notebook. It intentionally does not track raw retinal images,
generated caches, model checkpoints, predictions, or analysis outputs.

Exact numerical reproduction requires the same source image dataset. The
manifests validate paths and split membership but cannot verify private source
image bytes that are not distributed with the repository.

The primary claim is deliberately limited to the fixed Color4-42 and independent
RGB4 ensembles defined above. Only RGB was trained with the four additional
control seeds; the repository does not claim seed-population robustness for
Lab, HSV, or YCrCb.
