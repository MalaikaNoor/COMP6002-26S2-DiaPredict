"""Reproduce DiaPredict Section 5 figures from repository CSVs.
Run: python generate_figures.py
Dependencies: pandas, numpy, matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'models'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 120, 'savefig.dpi': 300})


def read(filename):
    df = pd.read_csv(DATA / filename)
    required = {'Seed', 'Dataset', 'Approach', 'ROC-AUC'}
    assert required.issubset(df.columns), f'Missing columns in {filename}'
    return df


def save(fig, stem):
    fig.savefig(OUT / (stem + '.png'), bbox_inches='tight')
    fig.savefig(OUT / (stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def comparison_plot(data, datasets, approaches, title, stem, ylim):
    fig, ax = plt.subplots(figsize=(9, 4.7))
    xs = np.arange(len(approaches))
    offsets = np.linspace(-0.15, 0.15, len(datasets))
    for i, dataset in enumerate(datasets):
        means, stds = [], []
        for label in approaches:
            vals = data[(data.Dataset == dataset) & (data.Label == label)]['ROC-AUC']
            assert len(vals) == 5, f'Expected five seeds: {dataset}, {label}; got {len(vals)}'
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        ax.errorbar(xs + offsets[i], means, yerr=stds, marker='o', linestyle='none',
                    capsize=4, label=dataset, markersize=6)
    ax.set_xticks(xs, approaches)
    ax.set_ylabel('Test ROC-AUC (mean ± SD across five seeds)')
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, stem)
    print(stem, 'saved')


sources = [
    ('prototype_v2_multiseed_results.csv', 'Federated', 'V2'),
    ('prototype_v3_multiseed_results.csv', 'Federated V3', 'V3'),
    ('prototype_contrastive_multiseed_results.csv', 'Federated Contrastive', 'V4'),
    ('prototype_coral_multiseed_results.csv', 'Federated Coral', 'V5'),
]
original = []
local = read(sources[0][0]); local = local[local.Approach == 'Local'].copy(); local['Label'] = 'Local'
original.append(local)
for filename, approach, label in sources:
    df = read(filename)
    df = df[df.Approach == approach].copy()
    df['Label'] = label
    original.append(df)
original = pd.concat(original, ignore_index=True)
comparison_plot(original, ['Dataset 1', 'Dataset 2'], ['Local', 'V2', 'V3', 'V4', 'V5'],
                'Original datasets: ROC-AUC by method', 'Figure_5_1_Original_ROCAUC', (0.93, 0.985))

additional = read('additional_pair_multiseed_results.csv')
additional['Label'] = additional.Approach.map({'Local': 'Local', 'Federated V3': 'V3', 'Federated V5': 'V5'})
assert additional.Label.notna().all()
comparison_plot(additional, ['Pima', 'Early Stage'], ['Local', 'V3', 'V5'],
                'Additional datasets: ROC-AUC by method', 'Figure_5_2_Additional_ROCAUC', (0.5, 1.0))

for suffix, dataset in [('a', 'Pima'), ('b', 'Early Stage')]:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for label in ['Local', 'V3', 'V5']:
        rows = additional[
            (additional.Dataset == dataset) &
            (additional.Label == label)
        ].sort_values('Seed')

        assert len(rows) == 5
        ax.plot(rows.Seed, rows['ROC-AUC'], marker='o', label=label)

    ax.set_xticks([42, 52, 62, 72, 82])
    ax.set_xlabel('Training seed')
    ax.set_ylabel('Test ROC-AUC')
    ax.set_title(f'{dataset}: ROC-AUC across five training seeds')
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()

    save(fig, f'Figure_5_3{suffix}_Seed_Variability_{dataset.replace(" ", "_")}')
    print(f'Figure 5.3{suffix} — {dataset} seed variability saved')

print('All figures generated from raw repository result CSVs.')
