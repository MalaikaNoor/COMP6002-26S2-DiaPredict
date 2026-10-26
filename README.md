# COMP6002-26S2-DiaPredict — Heterogeneous Federated Learning for Diabetes Prediction

## Project Overview

DiaPredict is a COMP6002 Computer Science Project at Curtin University investigating federated learning for diabetes prediction across heterogeneous datasets.

The project explores whether models trained on datasets with different features can collaborate without sharing raw patient records. Several approaches were implemented and compared against local-only models.

## Repository Structure

- `data/` — Four diabetes datasets used in the experiments.
- `notebooks/` — Data exploration, federated learning prototypes and evaluation notebooks.
- `models/` — Saved model checkpoints and experimental results in CSV format.
- `visualisations/` — Figure-generation script, supporting result CSVs and evaluation figures.
- `requirements.txt` — Python dependencies.

## Installation

Clone the repository:

```bash
git clone https://github.com/MalaikaNoor/COMP6002-26S2-DiaPredict.git
cd COMP6002-26S2-DiaPredict
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

## Notebooks

The notebooks are numbered in their intended development sequence:

1. `01_Data_Exploration.ipynb`
2. `02_Dataset2_Exploration.ipynb`
3. `03_Federated_Prototype.ipynb`
4. `04_Heterogeneous_Federated_V2.ipynb`
5. `05_V2_Multiseed_Robustness.ipynb`
6. `06_V3_Latent_Alignment.ipynb`
7. `07_Contrastive_Alignment.ipynb`
8. `08_Coral_Alignment.ipynb`
9. `09_Additional_Datasets_Evaluation.ipynb`

Open the notebooks in Jupyter to inspect or rerun the experiments. The repository also contains saved results, so the evaluation figures can be regenerated without retraining the models.

## Experimental Evaluation

The project compares local-only prediction against heterogeneous federated learning approaches, including prototype alignment, contrastive alignment and CORAL.

Experiments were evaluated across multiple random seeds using ROC-AUC. Results for the original and additional datasets are stored in `models/`.

## Reproducing the Evaluation Figures

From the repository root, run:

```bash
cd visualisations
python generate_figures.py
```

This command regenerates:

- Figure 5.1 — ROC-AUC comparison on the original datasets.
- Figure 5.2 — ROC-AUC comparison on the additional datasets.
- Figure 5.3a — Seed variability on the Pima dataset.
- Figure 5.3b — Seed variability on the Early Stage Diabetes dataset.

The figures are saved in `visualisations/figures/` in PNG and PDF formats.

## Repository

https://github.com/MalaikaNoor/COMP6002-26S2-DiaPredict