<p align="center"><img src="reports/figures/banner.jpg" width="100%" alt="Urban Microclimate Modelling: neural surrogates of large-eddy simulation for pedestrian-level wind in unseen cities"></p>

[![CI](https://img.shields.io/github/actions/workflow/status/mahir-m01/Urban-Microclimate-Course-Project/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white&labelColor=24292f&style=flat-square)](https://github.com/mahir-m01/Urban-Microclimate-Course-Project/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white&labelColor=24292f&style=flat-square)](https://www.python.org/)
[![uv](https://img.shields.io/badge/env-uv-6340ac?logo=uv&logoColor=white&labelColor=24292f&style=flat-square)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/lint-ruff-c2410c?logo=ruff&logoColor=white&labelColor=24292f&style=flat-square)](https://github.com/astral-sh/ruff)
[![PyTorch](https://img.shields.io/badge/dl-PyTorch-ee4c2c?logo=pytorch&logoColor=white&labelColor=24292f&style=flat-square)](https://pytorch.org/)
[![Data: UrbanTALES](https://img.shields.io/badge/data-UrbanTALES-2a78d6?labelColor=24292f&style=flat-square)](https://urbantales.vercel.app)
[![Licence: CC BY-NC 4.0](https://img.shields.io/badge/licence-CC%20BY--NC%204.0-57606a?labelColor=24292f&style=flat-square)](https://creativecommons.org/licenses/by-nc/4.0/)
![Phase 1: machine learning](https://img.shields.io/badge/phase%201-machine%20learning-1a7f5a?labelColor=24292f&style=flat-square)
![Phase 2: deep learning](https://img.shields.io/badge/phase%202-deep%20learning-bf3989?labelColor=24292f&style=flat-square)

Surrogate models for pedestrian-level urban wind, trained on 538 large-eddy simulations of idealized building arrays and real neighbourhoods in 22 countries, and tested on cities they have never seen.

## Overview

Urban form controls how wind moves through streets. Deep street canyons trap heat and pollution, while open corridors and tall buildings create uncomfortable or unsafe gusts. Large-eddy simulation (LES) resolves these effects, but costs thousands of CPU-hours per neighbourhood: the 538 simulations used here took more than 3 million CPU-hours (Nazarian et al., 2025).

This project builds surrogates that predict the time-averaged wind 1.75 m above the ground from a building-height map and the wind direction alone. It has two phases on the same data, folds and metrics:

- **Phase 1, machine learning.** Physics-informed morphometric features and eight learners. They predict the mean wind of an unseen neighbourhood well, but only part of the wind map inside it.
- **Phase 2, deep learning.** A periodic U-Net that reads the building map directly, built to respect the periodicity and the exact symmetries of the simulations. It roughly doubles the per-neighbourhood accuracy of the Phase 1 wind map, carries calibrated uncertainty, and each of its networks predicts a whole neighbourhood in under a second.

Every score is measured on cities and layout families held out from training. The work is a study of surrogate accuracy and uncertainty, not a tool for safety-critical design approval.

**Reports:** [combined paper](reports/Combined_Phase1_Phase2_Paper.pdf) · [Phase 1 report](reports/Phase1_Report.pdf) · [Phase 2 report](reports/Phase2_Report.pdf) · [Phase 1 slides](reports/Phase1_Presentation.pdf) · [Phase 2 slides](reports/Phase2_Presentation.pdf)

## Results at a glance

Out-of-fold scores on the 482 development cases. Each case is predicted by models that never saw its city or layout family.

| Wind map at every open cell | Pooled $R^2$ | Median per-case $R^2$ (idealized / realistic) | MAE ($u_*$) |
| --- | --- | --- | --- |
| Flat map at the true case mean (oracle) | 0.311 | 0 / 0 | 0.816 |
| Phase 1: pixel-level LightGBM, 30 features | 0.591 | 0.418 / 0.247 | 0.611 |
| Phase 2: periodic U-Net, one network | 0.709 to 0.712 | 0.621 to 0.683 / 0.523 to 0.544 | 0.472 to 0.486 |
| **Phase 2: final U-Net, three networks averaged** | **0.734** | **0.747 / 0.573** | **0.450** |

- The mean wind of a neighbourhood is predicted from 34 features by LightGBM with $R^2$ = 0.876 (Phase 1).
- The final U-Net is more accurate than the Phase 1 wind map in 97.5% of cases and in all five folds.
- Its 90% intervals, calibrated on held-out cities, cover 0.899 of cells.
- On the 53 frozen test cases, scored once, it reaches a pooled $R^2$ of 0.840 and a median per-case $R^2$ of 0.655, against 0.680 and 0.351 for the Phase 1 pixel model, with the higher per-case $R^2$ in all 53 cases.

<p align="center"><img src="reports/figures/phase2_wind_maps.png" width="860" alt="Simulated wind, Phase 1 pixel model, final U-Net and the U-Net error for three held-out cases"></p>
<p align="center"><sub>Held-out cases ranked by the U-Net's per-case R²: the median idealized case, the median realistic case and a lower-quartile realistic case. Columns: the simulation, the Phase 1 pixel model, the final U-Net, and the U-Net error (blue too slow, red too fast).</sub></p>

## Dataset

This work uses UrbanTALES (Nazarian et al., 2025), a public collection of 538 PALM large-eddy simulations of neutral atmospheric flow over urban layouts.

| Subset | Cases | Description |
| --- | --- | --- |
| Idealized | 224 | Aligned and staggered building arrays; eight plan-area densities; uniform and variable heights; two wind directions; 0.5 m grid |
| Realistic | 314 | Neighbourhoods in 22 countries (50 city codes), derived from OpenStreetMap; uniform and real building heights; 1 m grid (0.5 m for 5 cases) |

Each case provides:
- a building-height raster;
- time-averaged pedestrian-level velocity and turbulence fields;
- horizontally averaged vertical profiles;
- the simulation configuration.

<p align="center"><img src="reports/figures/dataset_cases.png" width="820" alt="Three UrbanTALES cases: building heights and pedestrian-level wind"></p>
<p align="center"><sub>Three development cases: building heights (top) and the simulated wind at 1.75 m, normalised by the friction velocity (bottom). Jets form where streets align with the wind; wakes fill the spaces behind buildings.</sub></p>

The data is distributed by its authors under CC BY-NC 4.0. The simulation fields are not redistributed in this repository; `data/` holds only the case catalog, the official per-case metadata table and the name map between them.

**Data quality.** An audit of every file found four issues that change downstream results. All are handled once, in the shared loader `urbanwind.data.load_case`; details are in [`data/README.md`](data/README.md) and [`pipeline/S02_preprocessing`](pipeline/S02_preprocessing).
- The building rasters are stored upside down relative to the wind fields. Open-ground agreement is 0.71 as stored and 0.99 after a flip.
- Some files store values inside buildings, so open ground is taken from the raster, not from missing values.
- Some open-ground cells hold a time-mean speed of exactly zero: the simulation treated them as buildings. They are left out (284 of 292 realistic development cases have some). Together with the raster mask, this changes the mean wind of 59 development cases by more than 1%, by up to 14.6%.
- Each case has its own friction velocity $u_*$, which sets the wind scale.

## Pipeline

<p align="center"><img src="reports/figures/architecture.png" width="100%" alt="System architecture: UrbanTALES, loader and audit, grouped splits, Phase 1 features and models, Phase 2 cache and periodic U-Net, hybrids, uncertainty, grouped evaluator and frozen test set"></p>
<p align="center"><sub>One loader applies every data correction for both phases. Features never read a simulated field, all training and model selection use grouped folds of the development cases, and the frozen test groups enter only the final scoring.</sub></p>

| Stage | Phase | What it does |
| --- | --- | --- |
| [S01](pipeline/S01_data_ingestion) | Data | Case catalog and a restartable download of all 538 cases |
| [S02](pipeline/S02_preprocessing) | Data | Profile preprocessing and case-level targets |
| [S03](pipeline/S03_splits_evaluation) | Data | Grouped folds, frozen test groups, metrics and the evaluator |
| [S04](pipeline/S04_eda) | Data | Exploratory analysis and quality checks |
| [S05](pipeline/S05_feature_engineering) | 1 | 34 case-level and 30 pixel-level features: periodic morphology, Macdonald and Kanda roughness, fetch and sheltering |
| [S06](pipeline/S06_case_models) | 1 | Eight case-level models under nested grouped cross-validation, group bootstrap, SHAP |
| [S06b](pipeline/S06b_feature_selection_AS) | 1 | Feature selection and PCA |
| [S07](pipeline/S07_pixel_model) | 1 | Pixel-level LightGBM, the level-times-pattern hybrid and diagnostics |
| [S09](pipeline/S09_phase1_evaluation) | 1 | Phase 1 scoring of the frozen test set |
| [S10](pipeline/S10_dl_data_pipeline) | 2 | 1 m cache in a wind-aligned frame, wrap-around training patches, whole-domain prediction |
| [S11](pipeline/S11_unet_baseline) | 2 | Periodic U-Net, shared trainer and tuning; three seeds on five folds |
| [S12](pipeline/S12_architecture_search) | 2 | Residual, attention-gated and Fourier-operator networks at an equal budget; the loss library |
| [S13](pipeline/S13_hybrid) | 2 | Error analysis, hybrids with the Phase 1 models, a ConvNeXt U-Net, and the final 200-epoch three-seed model |
| [S14](pipeline/S14_dl_uncertainty) | 2 | Deep ensemble and MC dropout uncertainty, split-conformal intervals calibrated on held-out groups |
| [S15](pipeline/S15_ablations_validation) | 2 | One-at-a-time ablations of the network and its inference time |
| [S16](pipeline/S16_frozen_test) | 2 | Phase 2 scoring of the frozen test set, run once after every choice is fixed |

## Phase 1 results

The full write-up is the [Phase 1 report](reports/Phase1_Report.pdf), and [`reports/references.md`](reports/references.md) lists the papers it builds on. All development numbers are out-of-fold scores on 482 development cases. Each fold holds out whole cities and layout families (93 groups), and hyperparameters are chosen in an inner grouped loop. The numbers below follow the pipeline audit of 27 October 2026, which corrected the features, the targets and the development groups.

### Case level: mean wind of a neighbourhood

| Model | $U_\mathrm{mean}$ $R^2$ | idealized | realistic | $U_\mathrm{p95}$ $R^2$ |
| --- | --- | --- | --- | --- |
| LightGBM | **0.876** | 0.961 | **0.787** | **0.767** |
| Random forest | 0.828 | 0.958 | 0.694 | 0.725 |
| Gaussian process (ARD Matérn) | 0.816 | **0.976** | 0.655 | 0.710 |
| XGBoost | 0.784 | 0.956 | 0.608 | 0.723 |
| SVR (RBF) | 0.717 | 0.958 | 0.472 | 0.700 |
| kNN | 0.655 | 0.930 | 0.373 | 0.668 |
| Ridge | 0.197 | 0.864 | -0.480 | 0.380 |
| Baseline B1: random forest, 9 basic features | 0.742 | 0.881 | 0.587 | 0.660 |

- **LightGBM leads.** It beats the baseline by 0.135 R² [0.095, 0.178] (95% group-bootstrap interval), and the random forest by 0.048 [0.005, 0.099]. The gains of tuned XGBoost and the Gaussian process over the baseline are not significant.
- **Tuning can help or hurt.** With default settings XGBoost (0.851) and LightGBM (0.829) are tied. Tuning lifts LightGBM, but XGBoost's inner search picks settings that fail on two very open neighbourhoods whose cities are held out with them.
- **What drives the wind.** SHAP ranks distance to walls, frontal area, upwind and lateral fetch, and built volume highest.
- **Random splits flatter models.** Random 5-fold cross-validation overstates realistic $R^2$ by up to 0.04 for tree models and by 0.61 for ridge.

<p align="center"><img src="pipeline/S06_case_models/outputs/s06c/figures/model_ladder_u_mean.png" width="640" alt="Case-level model comparison"></p>

### Field level: the wind map

| Prediction at every open cell | Pooled $R^2$ | Median per-case $R^2$ (idealized / realistic) | Median NRMSE |
| --- | --- | --- | --- |
| True case mean everywhere (oracle) | 0.311 | 0 / 0 | 0.549 |
| Pixel-level LightGBM, 30 features | **0.591** | 0.418 / 0.247 | 0.468 |
| Hybrid: pattern times case-level mean | 0.576 | **0.431 / 0.251** | **0.467** |

- **Level versus pattern.** A tabular model recovers the level of each field, but only part of its pattern: corner jets, wakes and canyon vortices are smoothed, and peak winds are underpredicted.
- **The limit is the features.** A random case split changes this little. A far less regularised model fits its training cases better but scores no better on unseen ones.
- **Why Phase 2.** This gap is the motivation for convolutional models that read the geometry directly.

<p align="center"><img src="pipeline/S07_pixel_model/outputs/figures/error_maps_realistic.png" width="820" alt="Error maps on held-out realistic neighbourhoods"></p>

### Frozen test set

After all choices were fixed on the development folds, the final models were refitted on all 482 development cases and scored on the 53 cases of the 24 frozen test groups (34 idealized, 19 realistic). The test set was first scored on 26 October 2026 and scored a second time on 27 October 2026, after the pipeline audit; no test score was used for any choice, and `pipeline/S09_phase1_evaluation/outputs/test_scored.json` records both runs and the reason. Details are in [`pipeline/S09_phase1_evaluation`](pipeline/S09_phase1_evaluation).

| Model | $U_\mathrm{mean}$ $R^2$ | idealized | realistic |
| --- | --- | --- | --- |
| Gaussian process | 0.931 | 0.985 | 0.843 |
| XGBoost | 0.898 | 0.975 | 0.772 |
| Random forest | 0.883 | 0.971 | 0.740 |
| kNN | 0.870 | 0.959 | 0.723 |
| LightGBM | 0.865 | 0.974 | 0.690 |
| SVR (RBF) | 0.857 | 0.974 | 0.669 |
| Baseline B1 | 0.827 | 0.919 | 0.666 |
| Pixel-level LightGBM (pooled field $R^2$) | 0.680 | 0.893 | 0.617 |

- Every non-linear model beats the baseline, including kNN and SVR, which were below it in development, and the field models reproduce the development picture.
- The order of the leading models differs from development: with only 19 realistic test cases, a few cases decide it. The test set confirms generalisation but cannot rank close models; the development comparison over 482 cases is the evidence for ranking.

## Phase 2 results

The full write-up is the [Phase 2 report](reports/Phase2_Report.pdf). Every network is trained and scored on the same 482 development cases, five grouped folds and metrics as Phase 1, so the comparison is like for like.

### The network

- **Periodic U-Net.** Four levels, 4.86 M parameters. Every convolution and every upsampling wraps around the domain edges, because each simulated neighbourhood repeats periodically.
- **Inputs.** Building height, the building mask, the distance to the nearest wall, and the wind direction. Each case is turned so the wind blows within 0° to 45° of +x; mirroring across the wind is then an exact symmetry, used to augment training and averaged at prediction.
- **Training.** Masked L1 loss on open ground, settings tuned with Optuna on inner-validation groups. The final model averages three seeds trained for 200 epochs.

<p align="center"><img src="reports/figures/phase2_inputs.png" width="860" alt="Input representation of one case"></p>
<p align="center"><sub>One development case (Adelaide): the raster as released, the same case turned into the wind-aligned frame, the periodic distance to the nearest wall, and the target with one 256 m training patch.</sub></p>

<p align="center"><img src="reports/figures/phase2_unet.png" width="600" alt="Architecture of the periodic U-Net"></p>

### Development results

| Model | Pooled $R^2$ | idealized | realistic | Median per-case $R^2$ | idealized | realistic | MAE ($u_*$) | $\Delta U_\mathrm{p95}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flat map at the true case mean (oracle) | 0.311 | 0.394 | 0.298 | 0 | 0 | 0 | 0.816 | -1.745 |
| Phase 1 pixel-level LightGBM | 0.591 | 0.876 | 0.550 | 0.304 | 0.418 | 0.247 | 0.611 | -0.727 |
| U-Net, 100 epochs, seeds 0 to 2 | 0.709 to 0.712 | 0.935 to 0.944 | 0.677 to 0.680 | 0.560 to 0.595 | 0.621 to 0.683 | 0.523 to 0.544 | 0.472 to 0.486 | -0.457 to -0.439 |
| **Final: 200 epochs, three seeds averaged** | **0.734** | **0.953** | **0.702** | **0.629** | **0.747** | **0.573** | **0.450** | **-0.415** |

$\Delta U_\mathrm{p95}$ is the mean shift of each case's 95th-percentile speed; negative values mean underpredicted peaks. The final model's MAE is 0.093 m/s.

- **Far better than the features.** Against the Phase 1 pixel model, the final U-Net gains 0.143 [0.108, 0.191] in pooled $R^2$ and 0.324 [0.291, 0.351] in median per-case $R^2$ (95% intervals from resampling whole cities and layout families). It is better in 97.5% of cases and in all five folds.

<p align="center"><img src="reports/figures/phase2_per_case.png" width="440" alt="Per-case R² of the final U-Net against the Phase 1 pixel model"></p>

### Architectures

At an equal budget, residual and attention-gated U-Nets and a Fourier neural operator are all less accurate than the plain periodic U-Net, and a ConvNeXt U-Net screened on one fold scored below every U-Net seed ([S12](pipeline/S12_architecture_search), [S13](pipeline/S13_hybrid)).

<p align="center"><img src="pipeline/S12_architecture_search/outputs/figures/s12_comparison.png" width="760" alt="Architecture comparison against the U-Net"></p>

### Where the remaining error lies

Real neighbourhoods hold 97 to 98% of the squared error, and 85 to 89% of that is the pattern inside each case. Averaging over 16 m blocks barely raises the score, so what remains is the street-scale arrangement of jets and wakes, not fine detail. The network already predicts each case's mean speed about as well as the Phase 1 case-level model, so a hybrid that takes the level from Phase 1 adds nothing; stacking the Phase 1 features on the network's error improves peak speeds by about a fifth at a small cost in mean accuracy ([S13](pipeline/S13_hybrid)).

<p align="center"><img src="reports/figures/phase2_error_anatomy.png" width="860" alt="Where the error lies"></p>

### Ablations

Each ablation changes one setting of the network and is retrained and compared case by case with the unchanged network. Zero padding instead of wrap-around and training without the symmetry augmentation each cost about 0.03 in pooled $R^2$ over five folds, and a height-only input costs 0.044. A network trained on idealized arrays alone fails on real cities (realistic pooled $R^2$ of -0.035), and the score keeps rising with the number of training cities. Of the seven training losses tried on fold 0, plain L1 is best and only an added FFT term ties it. Details are in the [S15 summary](pipeline/S15_ablations_validation).

<p align="center"><img src="reports/figures/phase2_ablations.png" width="860" alt="Five-fold ablations of the network"></p>

### Uncertainty

The spread of the three networks is the uncertainty map. Split-conformal 90% intervals built from it, calibrated on held-out groups and separately for idealized and real cases, cover 0.899 of cells (0.900 idealized, 0.899 realistic) ([S14](pipeline/S14_dl_uncertainty)).

<p align="center"><img src="pipeline/S14_dl_uncertainty/figures/uncertainty_maps.png" width="760" alt="Prediction, error, ensemble spread and interval width for three cases"></p>

### Validation protocol

Letting neighbourhoods of a test city into training (a random case split, as in earlier work) raises the pooled score by about 0.03 once one outlying case is set aside, and on that case the model is confidently wrong: it learns that Melbourne is slow and misses the one fast Melbourne suburb by 3.4 $u_*$.

### Frozen test set

After every choice was fixed on the development folds, the final model was scored once, on 18 November 2026, on the 53 cases of the 24 frozen test groups (34 idealized, 19 realistic). Its 15 networks (three seeds on each of the five development folds) never saw a test case, so their predictions are averaged cell by cell, from the building raster and the wind direction only. The result tables were written a second time the same day with more decimals; the networks and predictions were unchanged, and `pipeline/S16_frozen_test/outputs/test_scored.json` records both runs and the reason. Details are in [`pipeline/S16_frozen_test`](pipeline/S16_frozen_test).

| Wind map at every open cell | Pooled $R^2$ (idealized / realistic) | Median per-case $R^2$ | MAE ($u_*$) | $\Delta P_{95}$ ($u_*$) |
| --- | --- | --- | --- | --- |
| Phase 1: pixel-level LightGBM | 0.680 (0.893 / 0.617) | 0.351 | 0.530 | -0.651 |
| **Phase 2: final U-Net** | **0.840 (0.960 / 0.805)** | **0.655** | **0.362** | **-0.353** |

- The U-Net has the higher per-case $R^2$ in all 53 test cases.
- Both models score higher than in development (0.734 and 0.591), partly because the test set is 64% idealized against 39% in development.
- Without the two test cases from cities that also have development neighbourhoods, the U-Net's pooled $R^2$ is 0.795.

## Using the trained model

The final Phase 2 model is the periodic U-Net of `pipeline/S13_hybrid/configs/unet_tuned_e200.yaml`, trained for 200 epochs with three seeds on each of the five development folds. `urbanwind.dl.predict` runs it on any building layout. It needs only the building heights and the wind direction, never a simulation.

To see it run, open [`demo/Phase2_Demo.ipynb`](demo/Phase2_Demo.ipynb): it predicts real neighbourhoods the model has never seen next to their simulations, the same neighbourhood with another wind direction, and a new layout with its uncertainty map, and summarises every setting of the model.

<p align="center"><img src="reports/figures/predict_flow.png" width="820" alt="How a prediction is made: layout, wind-aligned frame, input channels, 15 networks, back to the input grid, output"></p>

### 1. Get the weights

The trained weights (15 networks, 278 MB) are not in the repository; they are available from the authors on request. Place them as:

```text
results/checkpoints/s13_unet_tuned_e200_seed0/fold0.pt ... fold4.pt
results/checkpoints/s13_unet_tuned_e200_seed1/fold0.pt ... fold4.pt
results/checkpoints/s13_unet_tuned_e200_seed2/fold0.pt ... fold4.pt
```

Each file holds one network: one seed, trained on four of the five development folds.

### 2. Prepare the input

| Input | What it must be |
| --- | --- |
| Building heights | A 2D array in metres, 0 on open ground. Rows run along +y and columns along +x. File formats: `.npy`, `.csv`, `.txt`, or `.npz` with an array named `H`. |
| Grid spacing | 1 m (the default) or 0.5 m (`--dx 0.5`). A 0.5 m raster is coarsened to 1 m for the network, and the prediction is returned on the 0.5 m grid. |
| Wind direction | The direction the wind blows **towards**, in degrees from +x (columns) towards +y (rows): 0 blows towards +x, 90 towards +y. If +y is north and the wind comes **from** a compass bearing φ, use (270 − φ) mod 360; a westerly (φ = 270) gives 0. |
| Periodic domain | The simulations repeat periodically, and so does the network: the right edge continues into the left and the top into the bottom. If the edges of your raster do not join up, cells within about 130 m of an edge (half the network's 260 m receptive field) see the far side of the domain, so trust the interior. |

The whole raster is predicted in one pass, so the domain only has to fit in memory; the largest UrbanTALES domains have about a million cells.

### 3. Run it

```bash
uv run python -m urbanwind.dl.predict --height layout.npy --wind-dir 30 --out wind.npz --png wind.png
uv run python -m urbanwind.dl.predict --height layout.csv --dx 0.5 --wind-dir 225 --ustar 0.21 --out wind.npz
uv run python -m urbanwind.dl.predict --case AU-Ade-V1_d00 --folds 1 --out wind.npz   # an UrbanTALES case from data/raw
```

Or from Python:

```python
import numpy as np
from urbanwind.dl.predict import predict_layout

H = np.load("layout.npy")                          # metres, 0 on open ground
out = predict_layout(H, dx=1.0, wind_dir_deg=30)   # averages all 15 networks
wind, spread = out["U"], out["spread"]             # U/u* and its spread, NaN on buildings
```

- By default all 15 networks are averaged; `--seeds` and `--folds` choose a subset. For a layout outside the dataset, every network is out of sample.
- For an UrbanTALES development case, use only the networks of its own fold (`--folds k`, the case's fold in `data/splits.json`); the other folds were trained on it. Run this way, the command reproduces the reported out-of-fold predictions.
- One network predicts a whole neighbourhood in a median of 0.10 s (idealized arrays) to 0.23 s (real neighbourhoods) on an Apple M3 Pro GPU. It also runs on a CPU, more slowly.

### 4. Read the output

`wind.npz` holds:

| Array | Meaning |
| --- | --- |
| `U` | Time-averaged wind speed 1.75 m above the ground, divided by the friction velocity u*, on the input grid. NaN on buildings. |
| `spread` | Standard deviation of `U` across the networks: high where they disagree, mostly along jets and around corners. |
| `building` | True on building cells. |
| `U_ms` | `U` in m/s, only when `--ustar` is given. |
| `n_networks`, `wind_dir_deg`, `dx` | The settings used. |

<p align="center"><img src="reports/figures/predict_example.png" width="760" alt="Example output of the predict command"></p>
<p align="center"><sub>The <code>--png</code> output for a development case: U/u* (left) and the spread across the networks (right), buildings in grey.</sub></p>

- **Units.** The simulations are driven by a pressure gradient, so the natural output is U/u*. In UrbanTALES u* lies between 0.2015 and 0.2106 m/s. Compare layouts in U/u* directly; for m/s, multiply by the u* of the forcing you assume (`--ustar`).
- **Accuracy.** On the 482 development cases, each predicted by networks that never saw its city or layout family, the three-seed average reaches a pooled R² of 0.734 and a median per-case R² of 0.629 (idealized 0.747, realistic 0.573), with a mean absolute error of 0.450 u* (0.093 m/s). It underpredicts the fastest winds: the 95th percentile by about 0.4 u* on average. Errors are largest along the strongest jets and within 1 m of walls.
- **Uncertainty.** `spread` is a relative guide to where the prediction is less certain. Turning it into calibrated 90% intervals needs a scale factor fitted on held-out folds; `pipeline/S14_dl_uncertainty` does this for the development cases.
- **Limits.** The model inherits the simulations' assumptions: neutral, isothermal flow at one forcing, buildings as solid blocks without trees or terrain, and one height above the ground. The training layouts span plan-area densities of 0.06 to 0.64; layouts far outside that range are extrapolation.

## Reproducing the results

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv). Phase 2 training needs a GPU.

```bash
uv sync                                                          # pinned environment, including PyTorch
uv run python pipeline/S01_data_ingestion/download_urbantales.py   # case files into data/raw/ (about 4.8 GB; restartable)
uv run ruff check src pipeline && uv run pytest -q                 # lint and tests
```

Phase 1:

```bash
uv run python pipeline/S02_preprocessing/build_case_targets.py     # case-level targets (development cases only)
uv run python pipeline/S05_feature_engineering/process_cases.py    # case-level features
uv run python pipeline/S06_case_models/s06c_model_comparison.py    # case-level models (about 20 minutes)
PYTHONPATH=. uv run python pipeline/S06b_feature_selection_AS/run_feature_selection.py
PYTHONPATH=. uv run python pipeline/S06b_feature_selection_AS/run_pca.py
PYTHONPATH=. uv run python pipeline/S06b_feature_selection_AS/plot_and_consolidate.py
uv run python pipeline/S07_pixel_model/s07_pixel_model.py         # pixel-level model (about 35 minutes)
uv run python pipeline/S07_pixel_model/s07_diagnostics.py          # diagnostics (about 30 minutes)
```

Phase 2:

```bash
uv run python pipeline/S10_dl_data_pipeline/build_cache.py         # deep-learning cache (under a minute)
uv run python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet_tuned.yaml --seed 0   # one network, all five folds
uv run python -m urbanwind.dl.train --config pipeline/S13_hybrid/configs/unet_tuned_e200.yaml --seed 0     # final recipe; repeat with --seed 1 and 2
uv run python pipeline/S11_unet_baseline/s11_results.py             # S11 tables and figures
uv run python pipeline/S12_architecture_search/s12_results.py       # architecture comparison
uv run python -m pipeline.S13_hybrid.s13_final                     # three-seed ensembles
uv run python pipeline/S14_dl_uncertainty/run_s14_pipeline.py       # uncertainty and intervals
uv run python -m pipeline.S15_ablations_validation.s15_results  # ablation tables, after training the S15 configs
uv run python pipeline/S16_frozen_test/s16_score.py                # frozen test set, scored once
```

- The splits in `data/splits.json` are frozen; do not regenerate them.
- One fold of the 100-epoch U-Net takes about an hour on a Kaggle T4 GPU. Each training records the commit of its code and a hash of its settings, and refuses to resume with changed settings.
- The evaluator rejects predictions for test cases outside the final scoring, and the test-scoring scripts of both phases (S09, S16) refuse a second run unless the reason is recorded.
- Each stage README lists its inputs, outputs, results and the exact commands.

## Repository structure

```text
Urban-Microclimate-Course-Project/
├── data/                           case catalog, metadata, name map, frozen splits
├── demo/                           Phase 2 demo: the trained U-Net on real and new layouts
├── src/urbanwind/                  shared package: loader, metrics, evaluator, splits, features
│   └── dl/                         deep learning: cache, patches, networks, losses, trainer, uncertainty, predict
├── pipeline/
│   ├── S01_data_ingestion/         catalog builder and downloader
│   ├── S02_preprocessing/          profile preprocessing and case-level targets
│   ├── S03_splits_evaluation/      grouped splits and evaluation protocol
│   ├── S04_eda/                    exploratory analysis and quality checks
│   ├── S05_feature_engineering/    case- and pixel-level features
│   ├── S06_case_models/            case-level models (S06a first run, S06c comparison)
│   ├── S06b_feature_selection_AS/  feature selection and PCA study
│   ├── S07_pixel_model/            pixel-level model and diagnostics
│   ├── S09_phase1_evaluation/      Phase 1 scoring of the frozen test set
│   ├── S10_dl_data_pipeline/       deep-learning cache, training patches and whole-domain prediction
│   ├── S11_unet_baseline/          periodic U-Net, trainer runs, tuning and results
│   ├── S12_architecture_search/    alternative architectures and the loss library
│   ├── S13_hybrid/                 error analysis, hybrids and the final model
│   ├── S14_dl_uncertainty/         ensemble and MC dropout uncertainty, conformal intervals
│   ├── S15_ablations_validation/   one-at-a-time ablations and inference time
│   └── S16_frozen_test/            Phase 2 scoring of the frozen test set
├── reports/                        reports, slides and the combined paper (PDF), README figures, key references
├── results/                        experiment log (runs.csv; runs_v1.csv is the log before the metric columns were corrected); predictions and weights are not tracked
├── .github/workflows/              lint and tests on every pull request
├── pyproject.toml, uv.lock         pinned environment
└── README.md
```

Each stage folder contains its scripts or notebooks, a stage summary and the outputs that later stages build on.

## Team

| Member | Phase 1 | Phase 2 |
| --- | --- | --- |
| Mahir Abdullah | Pipeline design and coordination; audit of all 538 cases, shared loader and evaluation tooling; case-level model comparison, pixel-level model, frozen-test evaluation | Deep-learning data pipeline, periodic U-Net, trainer and tuning; final model, error analysis and hybrids; integration and reports |
| Neeraj Singh | Exploratory analysis of profiles, rasters, wind fields, forcing and time series; master case table | Ablation studies and technical validation |
| Om Mishra | Profile preprocessing; grouped frozen splits, evaluation metrics and the evaluator | Loss-function library |
| Aparna Singh | Physics-informed feature library; first nested tuning of the case-level models; feature-selection and PCA study | Residual, attention-gated and Fourier-operator networks and their comparison |
| Ayush Aryan | First case-level model ladder and the feature-set ablation | Uncertainty: deep ensembles, MC dropout and conformal calibration |

## Citation

This project builds on the UrbanTALES dataset:

> Nazarian, N., Lu, J., Lipson, M., Hart, M. A., Liu, S., Krayenhoff, E. S., Blunn, L., and Martilli, A. (2025). UrbanTALES: A Large-Eddy Simulation Dataset for Urban Canopy Layer Turbulence and Parameterization. *Bulletin of the American Meteorological Society*, 106(12), E2461–E2478. https://doi.org/10.1175/BAMS-D-25-0061.1

Related work:

> Lu, J., Li, W., Hobeichi, S., Azad, S. A., and Nazarian, N. (2025). Machine learning predicts pedestrian wind flow from urban morphology and prevailing wind direction. *Environmental Research Letters*, 20, 054006. https://doi.org/10.1088/1748-9326/adc148

> Kanda, M., Inagaki, A., Miyamoto, T., Gryschka, M., and Raasch, S. (2013). A New Aerodynamic Parametrization for Real Urban Surfaces. *Boundary-Layer Meteorology*, 148, 357–377. https://doi.org/10.1007/s10546-013-9818-x

## Acknowledgements

Developed as a semester project in Advanced Machine Learning and Deep Learning. Simulation data courtesy of the UrbanTALES team at UNSW Sydney; GPU time from Kaggle.
