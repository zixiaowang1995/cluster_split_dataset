# Cluster-based Dataset Splitting Tool

## Introduction

This tool provides molecular clustering-based dataset splitting functionality, supporting three modes: random split, repeated random split, and cross-validation. It performs K-means clustering using ECFP4 molecular fingerprints to ensure similar molecules are assigned to the same cluster, then performs stratified splitting within each cluster to maintain class balance.

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

## Small Cluster Handling

The script automatically handles small clusters:

- Random split mode: Merges clusters with fewer than 10 samples into other clusters
- Cross-validation mode: Merges clusters with fewer samples than the number of folds into other clusters

## Intra-cluster Stratified Splitting Strategy

1. Within each cluster, separate positive and negative samples
2. Split positive and negative samples separately according to specified ratios
3. If a class has insufficient samples (fewer than 10 in random split mode, fewer than the number of folds in CV mode), perform random splitting on the entire cluster

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