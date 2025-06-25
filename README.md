# Cluster-based Dataset Splitting Tool (No Stratification)

## Introduction

This tool provides molecular clustering-based dataset splitting functionality, supporting three modes: random split, repeated random split, and cross-validation. It performs K-means clustering using ECFP4 molecular fingerprints to ensure similar molecules are assigned to the same cluster, then performs **random splitting within each cluster without considering label stratification**.

## Installation

```bash
pip install pandas numpy scikit-learn rdkit
```

## Usage

### 1. Random Split

```bash
python cluster_split_dataset.py input.csv output_dir --mode random --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### 2. Repeated Random Split

```bash
python cluster_split_dataset.py input.csv output_dir --mode random --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --n_repeats 5
```

### 3. Cross-validation

```bash
python cluster_split_dataset.py input.csv output_dir --mode cv --n_folds 5
```

## Parameters

### Basic Parameters
- `input_csv`: Input CSV file path
- `output_dir`: Output directory path
- `--mode`: Split mode, choose `random` or `cv`

### Clustering Parameters
- `--n_clusters`: Number of clusters (default: 5)

### Random Split Parameters
- `--train_ratio`: Training set ratio (default: 0.8)
- `--val_ratio`: Validation set ratio (default: 0.1)
- `--test_ratio`: Test set ratio (default: 0.1)
- `--n_repeats`: Number of repeated splits (default: 1)

### Cross-validation Parameters
- `--n_folds`: Number of CV folds (default: 5)

### Other Parameters
- `--random_seed`: Random seed (default: 42)

## Input File Format

The input CSV file must contain the following columns:
- `SMILES`: SMILES string of the molecule
- `label`: Label of the molecule (0 or 1)

Example:
```csv
SMILES,label
CCO,1
CCC,0
CCCC,1
```

## Output Files

### Random Split Mode
- `X_train.csv`: Training set
- `X_val.csv`: Validation set
- `X_test.csv`: Test set

### Repeated Random Split Mode
- `X_train1.csv`, `X_train2.csv`, ...: Training sets from multiple splits
- `X_val1.csv`, `X_val2.csv`, ...: Validation sets from multiple splits
- `X_test1.csv`, `X_test2.csv`, ...: Test sets from multiple splits

### Cross-validation Mode
- `X_train1.csv`, `X_train2.csv`, ...: Training sets for each fold
- `X_val1.csv`, `X_val2.csv`, ...: Validation sets for each fold

## Detailed Process Output

The script provides detailed output showing the splitting process:

### Cross-validation Mode Output Example:
```
=== 5-Fold Cross-Validation Mode ===
Total dataset size: 2476 compounds
Number of clusters: 30
Number of folds: 5
Minimum cluster size threshold: 5

--- Large Clusters Processing ---
Found 25 large clusters (size >= 5)

Cluster 0: 156 compounds
  Base size per fold: 31
  Remainder: 1
  Collected 1 remainder compounds

Cluster 1: 89 compounds
  Base size per fold: 17
  Remainder: 4
  Collected 4 remainder compounds

...

Total large cluster samples: 2340
Total remainders from large clusters: 15

--- Small Clusters Processing ---
Found 1 merged cluster (combining all small clusters): 136 compounds
  Added to remainder pool

--- Combined Remainder Processing ---
Combined remainder data: 151 samples
  Base size per fold: 30
  Final remainder: 1
  Final 1 samples randomly assigned to folds: [3]

--- Final Fold Generation ---
Fold 1: Training set: 1980 samples, Validation set: 496 samples
Fold 2: Training set: 1980 samples, Validation set: 496 samples
Fold 3: Training set: 1981 samples, Validation set: 495 samples
Fold 4: Training set: 1980 samples, Validation set: 496 samples
Fold 5: Training set: 1980 samples, Validation set: 496 samples
```

### Random Split Mode Output Example:
```
=== Random Split Mode (1 repeats) ===
Total dataset size: 2476 compounds
Number of clusters: 30
Split ratios - Train: 0.8, Val: 0.1, Test: 0.1
Minimum cluster size threshold: 10

--- Split 1 (random seed: 42) ---

--- Large Clusters Processing ---
Found 20 large clusters (size >= 10)

Cluster 0: 156 compounds
  Train: 124, Val: 15, Test: 15
  Total allocated: 154, Remainder: 2
  Collected 2 remainder compounds

Cluster 1: 89 compounds
  Train: 71, Val: 8, Test: 8
  Total allocated: 87, Remainder: 2
  Collected 2 remainder compounds

...

Total large cluster samples: 2100
Total remainders from large clusters: 25

--- Small Clusters Processing ---
Found 1 merged cluster (combining all small clusters): 376 compounds
  Added to remainder pool

--- Combined Remainder Processing ---
Combined remainder data: 401 samples
  Train: 320, Val: 40, Test: 40
  Total allocated: 400, Final remainder: 1
  Final 1 samples randomly assigned to: ['train']

--- Final Set Generation ---
Final results:
  Training set: 2000 samples
  Validation set: 247 samples
  Test set: 229 samples
```

## Small Cluster Handling (No Stratification)

- **Random split mode**: All clusters with fewer than 10 samples are merged into a single group, which is then split randomly according to the specified ratios.
- **Cross-validation mode**: All clusters with fewer samples than the number of folds are merged into a single group, which is then split randomly among the folds.

## Splitting Strategy (No Stratification)

1. **Clustering**: Molecules are clustered using K-means on ECFP4 fingerprints.
2. **Small Cluster Merging**: All clusters smaller than the threshold (10 for random split, n_folds for CV) are merged into a single group.
3. **Two-Stage Random Splitting**:
   - **Stage 1**: For each large cluster, samples are randomly assigned to each set/fold according to the specified ratio, regardless of label. Remainders from each cluster are collected.
   - **Stage 2**: All remainders from large clusters + merged small cluster group are combined and randomly split according to the same ratio.
   - **Final remainder**: If there are still leftover samples, they are randomly assigned to sets/folds.
4. **No stratification**: The splitting process does **not** consider label distribution.

## Examples

### Random Split Example

```bash
python cluster_split_dataset.py molecules.csv ./split_results --mode random --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --n_clusters 8 --random_seed 42
```

### 5-Fold Cross-validation Example

```bash
python cluster_split_dataset.py molecules.csv ./cv_results --mode cv --n_folds 5 --n_clusters 8 --random_seed 42
```

### Repeated Random Split Example

```bash
python cluster_split_dataset.py molecules.csv ./repeat_results --mode random --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --n_clusters 8 --random_seed 42 --n_repeats 3
```

## Notes

1. Ensure the input CSV file contains required columns (SMILES and label)
2. Labels must be binary classification (0 or 1)
3. In random split mode, train_ratio + val_ratio + test_ratio must equal 1.0
4. For small datasets, consider reducing the number of clusters
5. Random seed only affects data splitting, not clustering results (clustering always uses fixed seed 42)
6. **No stratification**: All splits are random within clusters, regardless of label distribution.
7. **Detailed output**: The script provides comprehensive information about the splitting process, including cluster sizes, allocation details, and remainder handling.