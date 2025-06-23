# -*- coding: utf-8 -*-
"""Cluster-based dataset splitting script
Supports three modes: random split, cross-validation, repeated random split
Performs clustering based on ECFP4 molecular fingerprints, then stratified split within each cluster
"""

import pandas as pd
import numpy as np
import argparse
import os
import random
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def calculate_ecfp4_fingerprints(smiles_list: List[str]) -> np.ndarray:
    """
    Calculate ECFP4 molecular fingerprints
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Fingerprint matrix (n_molecules, n_bits)
    """
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # ECFP4 (radius=2, equivalent to diameter 4)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            # Convert to numpy array
            fp_array = np.zeros((2048,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
            fingerprints.append(fp_array)
        else:
            # Invalid SMILES, use zero vector
            fingerprints.append(np.zeros(2048))
    
    return np.array(fingerprints)


def perform_clustering(fingerprints: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """
    Perform K-means clustering based on ECFP4 fingerprints
    
    Args:
        fingerprints: Molecular fingerprint matrix
        n_clusters: Number of clusters
        random_state: Random seed (for clustering only)
        
    Returns:
        Cluster label array
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(fingerprints)
    return cluster_labels


def merge_small_clusters(df: pd.DataFrame, cluster_col: str, min_cluster_size: int) -> pd.DataFrame:
    """
    Merge small clusters
    
    Args:
        df: DataFrame
        cluster_col: Cluster column name
        min_cluster_size: Minimum cluster size
        
    Returns:
        DataFrame with merged clusters
    """
    df_copy = df.copy()
    cluster_counts = df_copy[cluster_col].value_counts()
    
    # Identify small and large clusters
    small_clusters = cluster_counts[cluster_counts < min_cluster_size].index.tolist()
    large_clusters = cluster_counts[cluster_counts >= min_cluster_size].index.tolist()
    
    if not small_clusters:
        return df_copy
    
    print(f"Found {len(small_clusters)} small clusters (size < {min_cluster_size}), merging...")
    
    # If no large clusters, merge all small clusters into one
    if not large_clusters:
        print("All clusters are small, merging into one cluster")
        df_copy.loc[df_copy[cluster_col].isin(small_clusters), cluster_col] = 0
        return df_copy
    
    # Randomly assign small clusters to large clusters
    for small_cluster in small_clusters:
        # Randomly select a large cluster for merging
        target_cluster = random.choice(large_clusters)
        df_copy.loc[df_copy[cluster_col] == small_cluster, cluster_col] = target_cluster
        print(f"Merged cluster {small_cluster} into cluster {target_cluster}")
    
    return df_copy


def stratified_split_within_cluster(cluster_data: pd.DataFrame, 
                                   test_size: float, 
                                   val_size: float = None,
                                   min_samples_per_class: int = 10,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified split within cluster (using exact sample count calculation)
    
    Args:
        cluster_data: Data within cluster
        test_size: Test set ratio
        val_size: Validation set ratio (if None, no validation set)
        min_samples_per_class: Minimum samples per class
        random_state: Random seed
        
    Returns:
        (train_set, val_set, test_set) or (train_set, None, test_set)
    """
    labels = cluster_data['label'].values
    unique_labels = np.unique(labels)
    n_samples = len(cluster_data)
    
    # Check if each class has sufficient samples
    label_counts = pd.Series(labels).value_counts()
    insufficient_classes = label_counts[label_counts < min_samples_per_class]
    
    # Calculate exact sample counts
    if val_size is not None:
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # Handle remaining samples (randomly assign to sets)
        remaining = n_samples - (n_train + n_val + n_test)
        if remaining > 0:
            np.random.seed(random_state)
            # Randomly decide which set gets remaining samples
            choices = ['train', 'val', 'test']
            for _ in range(remaining):
                choice = np.random.choice(choices)
                if choice == 'train':
                    n_train += 1
                elif choice == 'val':
                    n_val += 1
                else:
                    n_test += 1
    else:
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        n_val = 0
    
    if len(insufficient_classes) > 0:
        print(f"  Insufficient samples for some classes (< {min_samples_per_class}), using random split")
        # Random split (using exact sample counts)
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        if val_size is not None:
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
            return (cluster_data.iloc[train_idx].copy(), 
                   cluster_data.iloc[val_idx].copy(), 
                   cluster_data.iloc[test_idx].copy())
        else:
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            return (cluster_data.iloc[train_idx].copy(), 
                   None, 
                   cluster_data.iloc[test_idx].copy())
    
    # 分层划分（尽量保持类别比例）
    try:
        if val_size is not None:
            # First split out training set
            train_data, temp_data = train_test_split(
                cluster_data, train_size=n_train,
                stratify=labels, random_state=random_state, shuffle=True
            )
            
            # Then split validation and test sets from remaining data
            temp_labels = temp_data['label'].values
            if len(temp_data) == n_val + n_test:
                val_data, test_data = train_test_split(
                    temp_data, train_size=n_val,
                    stratify=temp_labels, random_state=random_state, shuffle=True
                )
            else:
                # If sample count mismatch, use random split
                temp_indices = np.arange(len(temp_data))
                np.random.seed(random_state)
                np.random.shuffle(temp_indices)
                val_data = temp_data.iloc[temp_indices[:n_val]].copy()
                test_data = temp_data.iloc[temp_indices[n_val:]].copy()
            
            return train_data, val_data, test_data
        else:
            # Binary split: train/test
            train_data, test_data = train_test_split(
                cluster_data, train_size=n_train,
                stratify=labels, random_state=random_state, shuffle=True
            )
            return train_data, None, test_data
    except ValueError as e:
        print(f"  Stratified split failed, using random split: {e}")
        # Fallback to random split (using exact sample counts)
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        if val_size is not None:
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
            return (cluster_data.iloc[train_idx].copy(), 
                   cluster_data.iloc[val_idx].copy(), 
                   cluster_data.iloc[test_idx].copy())
        else:
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            return (cluster_data.iloc[train_idx].copy(), 
                   None, 
                   cluster_data.iloc[test_idx].copy())


def random_split_mode(df: pd.DataFrame, 
                     n_clusters: int,
                     train_ratio: float,
                     val_ratio: float,
                     test_ratio: float,
                     output_dir: str,
                     random_seed: int = 42,
                     n_repeats: int = 1) -> None:
    """
    Random split mode (supports repeated splits)
    
    Args:
        df: Input DataFrame
        n_clusters: Number of clusters
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        output_dir: Output directory
        random_seed: Initial random seed
        n_repeats: Number of repeats
    """
    print(f"\n=== Random Split Mode ({n_repeats} repeats) ===")
    print(f"Split ratios - Train: {train_ratio:.1f}, Val: {val_ratio:.1f}, Test: {test_ratio:.1f}")
    
    # Calculate ECFP4 fingerprints
    print("Calculating ECFP4 molecular fingerprints...")
    fingerprints = calculate_ecfp4_fingerprints(df['SMILES'].tolist())
    
    # Clustering (cluster only once)
    print(f"Performing K-means clustering (n_clusters={n_clusters})...")
    cluster_labels = perform_clustering(fingerprints, n_clusters, random_state=42)
    df['cluster'] = cluster_labels
    
    # Merge small clusters
    min_cluster_size = 10  # Minimum cluster size for random split mode
    df = merge_small_clusters(df, 'cluster', min_cluster_size)
    
    # Repeated splits
    for repeat in range(n_repeats):
        current_seed = random_seed + repeat
        print(f"\n--- Split {repeat + 1} (random seed: {current_seed}) ---")
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        unique_clusters = df['cluster'].unique()
        for cluster_id in unique_clusters:
            cluster_data = df[df['cluster'] == cluster_id].copy()
            print(f"Processing cluster {cluster_id}: {len(cluster_data)} samples")
            
            # Stratified split within cluster
            train_data, val_data, test_data = stratified_split_within_cluster(
                cluster_data, 
                test_size=test_ratio,
                val_size=val_ratio,
                min_samples_per_class=10,
                random_state=current_seed
            )
            
            train_dfs.append(train_data)
            if val_data is not None:
                val_dfs.append(val_data)
            test_dfs.append(test_data)
        
        # Merge results from all clusters
        final_train = pd.concat(train_dfs, ignore_index=True)
        final_test = pd.concat(test_dfs, ignore_index=True)
        final_val = pd.concat(val_dfs, ignore_index=True) if val_dfs else None
        
        # Shuffle indices (maintain compound-label correspondence)
        np.random.seed(current_seed)
        
        # Shuffle training set
        train_indices = np.arange(len(final_train))
        np.random.shuffle(train_indices)
        final_train = final_train.iloc[train_indices].reset_index(drop=True)
        
        # Shuffle test set
        test_indices = np.arange(len(final_test))
        np.random.shuffle(test_indices)
        final_test = final_test.iloc[test_indices].reset_index(drop=True)
        
        # Shuffle validation set (if exists)
        if final_val is not None:
            val_indices = np.arange(len(final_val))
            np.random.shuffle(val_indices)
            final_val = final_val.iloc[val_indices].reset_index(drop=True)
        
        # Save results
        suffix = f"{repeat + 1}" if n_repeats > 1 else ""
        final_train.to_csv(os.path.join(output_dir, f"X_train{suffix}.csv"), index=False)
        final_test.to_csv(os.path.join(output_dir, f"X_test{suffix}.csv"), index=False)
        if final_val is not None:
            final_val.to_csv(os.path.join(output_dir, f"X_val{suffix}.csv"), index=False)
        
        # Output statistics
        print(f"Training set: {len(final_train)} samples")
        print(f"Test set: {len(final_test)} samples")
        if final_val is not None:
            print(f"Validation set: {len(final_val)} samples")


def cross_validation_mode(df: pd.DataFrame,
                         n_clusters: int,
                         n_folds: int,
                         output_dir: str,
                         random_seed: int = 42) -> None:
    """
    Cross-validation mode
    
    Args:
        df: Input DataFrame
        n_clusters: Number of clusters
        n_folds: Number of folds
        output_dir: Output directory
        random_seed: Random seed
    """
    print(f"\n=== {n_folds}-Fold Cross-Validation Mode ===")
    
    # Calculate ECFP4 fingerprints
    print("Calculating ECFP4 molecular fingerprints...")
    fingerprints = calculate_ecfp4_fingerprints(df['SMILES'].tolist())
    
    # Clustering
    print(f"Performing K-means clustering (n_clusters={n_clusters})...")
    cluster_labels = perform_clustering(fingerprints, n_clusters, random_state=42)
    df['cluster'] = cluster_labels
    
    # Merge small clusters
    min_cluster_size = n_folds  # Minimum cluster size equals number of folds for CV mode
    df = merge_small_clusters(df, 'cluster', min_cluster_size)
    
    # Create cross-validation folds for each cluster
    unique_clusters = df['cluster'].unique()
    cluster_folds = {}
    
    for cluster_id in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_id].copy()
        labels = cluster_data['label'].values
        
        print(f"Cluster {cluster_id}: {len(cluster_data)} samples")
        
        # Check if stratified cross-validation is possible
        label_counts = pd.Series(labels).value_counts()
        min_count = label_counts.min()
        
        if min_count < n_folds:
            print(f"  Insufficient samples for some classes (< {n_folds}), using random CV")
            # Random cross-validation (using exact sample count division)
            indices = np.arange(len(cluster_data))
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            
            # Calculate exact sample count per fold
            n_samples = len(cluster_data)
            base_size = n_samples // n_folds
            remainder = n_samples % n_folds
            
            # Randomly decide which folds get extra samples (avoid always first few folds)
            if remainder > 0:
                np.random.seed(random_seed + cluster_id)  # Use cluster ID to ensure different allocation per cluster
                extra_folds = np.random.choice(n_folds, remainder, replace=False)
            else:
                extra_folds = []
            
            folds = []
            start_idx = 0
            for i in range(n_folds):
                # Randomly selected folds get one extra sample
                fold_size = base_size + (1 if i in extra_folds else 0)
                end_idx = start_idx + fold_size
                folds.append(indices[start_idx:end_idx])
                start_idx = end_idx
        else:
            # Stratified cross-validation
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            folds = [test_idx for _, test_idx in skf.split(cluster_data, labels)]
        
        cluster_folds[cluster_id] = (cluster_data, folds)
    
    # Generate training and validation sets for each fold
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1} ---")
        
        train_dfs = []
        val_dfs = []
        
        for cluster_id, (cluster_data, folds) in cluster_folds.items():
            # Current fold as validation set
            val_indices = folds[fold]
            # Other folds as training set
            train_indices = np.concatenate([folds[i] for i in range(n_folds) if i != fold])
            
            val_data = cluster_data.iloc[val_indices].copy()
            train_data = cluster_data.iloc[train_indices].copy()
            
            train_dfs.append(train_data)
            val_dfs.append(val_data)
        
        # Merge results from all clusters
        final_train = pd.concat(train_dfs, ignore_index=True)
        final_val = pd.concat(val_dfs, ignore_index=True)
        
        # Shuffle indices (maintain compound-label correspondence)
        np.random.seed(random_seed + fold)  # Use different random seed for each fold
        
        # Shuffle training set
        train_indices = np.arange(len(final_train))
        np.random.shuffle(train_indices)
        final_train = final_train.iloc[train_indices].reset_index(drop=True)
        
        # Shuffle validation set
        val_indices = np.arange(len(final_val))
        np.random.shuffle(val_indices)
        final_val = final_val.iloc[val_indices].reset_index(drop=True)
        
        # Save results
        final_train.to_csv(os.path.join(output_dir, f"X_train{fold + 1}.csv"), index=False)
        final_val.to_csv(os.path.join(output_dir, f"X_val{fold + 1}.csv"), index=False)
        
        # Output statistics
        print(f"Training set: {len(final_train)} samples")
        print(f"Validation set: {len(final_val)} samples")


def main():
    parser = argparse.ArgumentParser(description='Cluster-based dataset splitting script')
    
    # Basic parameters
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--mode', choices=['random', 'cv'], required=True,
                       help='Split mode: random (random split) or cv (cross-validation)')
    
    # Clustering parameters
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters (default: 5)')
    
    # Random split parameters
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of repeated splits (default: 1)')
    
    # Cross-validation parameters
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds (default: 5)')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file {args.input_csv} does not exist")
        return
    
    if args.mode == 'random':
        if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
            print("Error: Sum of train, validation, and test ratios must equal 1.0")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read data
    print(f"Reading data: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Validate required columns
    if 'SMILES' not in df.columns:
        print("Error: Input file must contain 'SMILES' column")
        return
    
    if 'label' not in df.columns:
        print("Error: Input file must contain 'label' column")
        return
    
    # Validate labels
    unique_labels = df['label'].unique()
    if not set(unique_labels).issubset({0, 1}):
        print(f"Error: Labels must be 0 or 1, found labels: {unique_labels}")
        return
    
    print(f"Dataset size: {len(df)} compounds")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Execute corresponding split mode
    if args.mode == 'random':
        random_split_mode(
            df=df,
            n_clusters=args.n_clusters,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            n_repeats=args.n_repeats
        )
    elif args.mode == 'cv':
        cross_validation_mode(
            df=df,
            n_clusters=args.n_clusters,
            n_folds=args.n_folds,
            output_dir=args.output_dir,
            random_seed=args.random_seed
        )
    
    print(f"\nSplitting completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()