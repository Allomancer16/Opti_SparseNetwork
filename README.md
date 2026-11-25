# Sparse Network Recovery from Gene Expression Data

Precision matrix estimation on TCGA Pan-Cancer RNA-Seq data (`801 × 20,531`) using Graphical Lasso.

## Project Structure

```text
Opti_SparseNetwork/
├── data/
│   ├── raw/                  # Raw UCI TCGA expression matrix & tumor labels
│   └── processed/            # Filtered & standardized data (X_std.npy)
├── src/
│   ├── load_data.py          # Load CSVs, filter top-variance genes, standardize
│   ├── easy_model.py         # GraphicalLasso / GraphicalLassoCV solver & plot
│   ├── verify.py             # Symmetry, PD, objective, and KKT verification
│   └── utils.py              # Shared matrix and numerical helpers
├── results/
│   ├── theta_easy.npy        # Recovered precision matrix
│   ├── metrics.json          # Verification metrics
│   └── figures/              # Network visualization (network_easy.png)
├── tests/
│   └── test_pipeline.py      # Unit & integration tests
├── main.py                   # End-to-end pipeline runner
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run End-to-End Pipeline
```bash
python3 main.py --n-genes 750 --lambda-grid 0.80 0.85 0.90 --cv 5 --out results
```

### Run Individual Steps
```bash
# 1. Load, filter top-variance genes, and standardize
python3 src/load_data.py --n-genes 750

# 2. Solve Graphical Lasso
python3 src/easy_model.py --input data/processed/X_std.npy --lambda-grid 0.85 --out results/theta_easy.npy

# 3. Verify solution (PD check, objective value, KKT stationarity)
python3 src/verify.py --theta results/theta_easy.npy --data data/processed/X_std.npy --lambda 0.85 --out results/metrics.json
```

### Run Tests
```bash
python3 -m pytest tests/test_pipeline.py -v
```
