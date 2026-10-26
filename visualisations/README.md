# DiaPredict reproducible Section 5 figures

These figures are generated from **unchanged `models/*multiseed_results.csv` files** in the uploaded DiaPredict repository.

## Run

```bash
pip install pandas numpy matplotlib
python generate_figures.py
```

Outputs are saved in `figures/` as 300-DPI PNG and vector PDF files.

- Figure 5.1: Original Dataset 1/2, Local and V2–V5 mean ROC-AUC with **sample standard deviation** error bars.
- Figure 5.2: Pima and Early Stage, Local/V3/V5 mean ROC-AUC with sample standard deviation error bars.
- Figure 5.3a and 5.3b: ROC-AUC for each training seed, separately for Pima and Early Stage.

Error bars describe variation across five training seeds on fixed data partitions; they are not confidence intervals or statistical significance tests.

The plotting script performs checks for five records per dataset/method. To add to your project repository, copy `generate_figures.py` and the `figures` directory into an appropriate location, or retain the script beside the `models` folder and adjust the `DATA` path.
