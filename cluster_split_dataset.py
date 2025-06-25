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


def merge_small_clusters_to_one(df: pd.DataFrame, cluster_col: str, min_cluster_size: int) -> pd.DataFrame:
    """
    Merge all clusters smaller than min_cluster_size into a single new cluster (-1)
    """
    df_copy = df.copy()
    cluster_counts = df_copy[cluster_col].value_counts()
    small_clusters = cluster_counts[cluster_counts < min_cluster_size].index.tolist()
    if not small_clusters:
        return df_copy
    print(f"Found {len(small_clusters)} small clusters (size < {min_cluster_size}), merging into one group (-1)...")
    df_copy.loc[df_copy[cluster_col].isin(small_clusters), cluster_col] = -1
    return df_copy


def random_split_indices(n, ratios, seed=42):
    """
    Randomly split n samples into groups according to ratios. Return a list of index arrays.
    """
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)
    sizes = [int(n * r) for r in ratios]
    total = sum(sizes)
    remain = n - total
    # 随机把余数分配到各组
    for i in np.random.choice(len(ratios), remain, replace=False):
        sizes[i] += 1
    splits = []
    start = 0
    for s in sizes:
        splits.append(indices[start:start+s])
        start += s
    return splits


def random_split_mode(df: pd.DataFrame, 
                     n_clusters: int,
                     train_ratio: float,
                     val_ratio: float,
                     test_ratio: float,
                     output_dir: str,
                     random_seed: int = 42,
                     n_repeats: int = 1) -> None:
    print(f"\n=== Random Split Mode ({n_repeats} repeats) ===")
    print(f"Total dataset size: {len(df)} compounds")
    print(f"Number of clusters: {n_clusters}")
    print(f"Split ratios - Train: {train_ratio:.1f}, Val: {val_ratio:.1f}, Test: {test_ratio:.1f}")
    print(f"Minimum cluster size threshold: 10")
    
    fingerprints = calculate_ecfp4_fingerprints(df['SMILES'].tolist())
    cluster_labels = perform_clustering(fingerprints, n_clusters, random_state=42)
    df['cluster'] = cluster_labels
    min_cluster_size = 10
    df = merge_small_clusters_to_one(df, 'cluster', min_cluster_size)
    
    for repeat in range(n_repeats):
        current_seed = random_seed + repeat
        print(f"\n--- Split {repeat + 1} (random seed: {current_seed}) ---")
        
        # 收集所有余数和合并族数据
        all_remainders = []
        unique_clusters = [c for c in df['cluster'].unique() if c != -1]
        
        print(f"\n--- Large Clusters Processing ---")
        print(f"Found {len(unique_clusters)} large clusters (size >= {min_cluster_size})")
        
        # 先分配满足阈值的大族
        train_dfs, val_dfs, test_dfs = [], [], []
        total_large_cluster_samples = 0
        total_remainders = 0
        
        for cluster_id in unique_clusters:
            cluster_data = df[df['cluster'] == cluster_id].copy()
            n = len(cluster_data)
            total_large_cluster_samples += n
            
            # 计算每个集合应该分到的样本数
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)
            total_allocated = n_train + n_val + n_test
            remainder = n - total_allocated
            
            print(f"\nCluster {cluster_id}: {n} compounds")
            print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
            print(f"  Total allocated: {total_allocated}, Remainder: {remainder}")
            
            # 先按基础大小分配
            train_data = cluster_data.iloc[:n_train]
            val_data = cluster_data.iloc[n_train:n_train+n_val]
            test_data = cluster_data.iloc[n_train+n_val:n_train+n_val+n_test]
            
            train_dfs.append(train_data)
            val_dfs.append(val_data)
            test_dfs.append(test_data)
            
            # 收集余数
            if remainder > 0:
                remainder_data = cluster_data.iloc[n_train+n_val+n_test:]
                all_remainders.append(remainder_data)
                total_remainders += remainder
                print(f"  Collected {remainder} remainder compounds")
        
        print(f"\nTotal large cluster samples: {total_large_cluster_samples}")
        print(f"Total remainders from large clusters: {total_remainders}")
        
        # 合并族数据
        merged_data = df[df['cluster'] == -1].copy()
        if len(merged_data) > 0:
            all_remainders.append(merged_data)
            print(f"\n--- Small Clusters Processing ---")
            print(f"Found 1 merged cluster (combining all small clusters): {len(merged_data)} compounds")
            print(f"  Added to remainder pool")
        
        # 将所有余数和合并族数据合并
        if all_remainders:
            combined_remainder_data = pd.concat(all_remainders, ignore_index=True)
            n_remainder = len(combined_remainder_data)
            print(f"\n--- Combined Remainder Processing ---")
            print(f"Combined remainder data: {n_remainder} samples")
            
            # 对合并的余数数据按比例分配
            n_train_remainder = int(n_remainder * train_ratio)
            n_val_remainder = int(n_remainder * val_ratio)
            n_test_remainder = int(n_remainder * test_ratio)
            total_allocated_remainder = n_train_remainder + n_val_remainder + n_test_remainder
            final_remainder = n_remainder - total_allocated_remainder
            
            print(f"  Train: {n_train_remainder}, Val: {n_val_remainder}, Test: {n_test_remainder}")
            print(f"  Total allocated: {total_allocated_remainder}, Final remainder: {final_remainder}")
            
            train_remainder = combined_remainder_data.iloc[:n_train_remainder]
            val_remainder = combined_remainder_data.iloc[n_train_remainder:n_train_remainder+n_val_remainder]
            test_remainder = combined_remainder_data.iloc[n_train_remainder+n_val_remainder:n_train_remainder+n_val_remainder+n_test_remainder]
            
            train_dfs.append(train_remainder)
            val_dfs.append(val_remainder)
            test_dfs.append(test_remainder)
            
            # 最终余数随机分配
            if final_remainder > 0:
                final_remainder_data = combined_remainder_data.iloc[n_train_remainder+n_val_remainder+n_test_remainder:]
                np.random.seed(current_seed)
                choices = ['train', 'val', 'test']
                assigned_to = []
                for i in range(final_remainder):
                    choice = np.random.choice(choices)
                    assigned_to.append(choice)
                    if choice == 'train':
                        train_dfs.append(final_remainder_data.iloc[[i]])
                    elif choice == 'val':
                        val_dfs.append(final_remainder_data.iloc[[i]])
                    else:
                        test_dfs.append(final_remainder_data.iloc[[i]])
                print(f"  Final {final_remainder} samples randomly assigned to: {assigned_to}")
        else:
            print(f"\nNo remainder data to process")
        
        print(f"\n--- Final Set Generation ---")
        
        final_train = pd.concat(train_dfs, ignore_index=True)
        final_val = pd.concat(val_dfs, ignore_index=True)
        final_test = pd.concat(test_dfs, ignore_index=True)
        
        # Shuffle
        for df_ in [final_train, final_val, final_test]:
            df_.reset_index(drop=True, inplace=True)
        
        suffix = f"{repeat + 1}" if n_repeats > 1 else ""
        final_train.to_csv(os.path.join(output_dir, f"X_train{suffix}.csv"), index=False)
        final_test.to_csv(os.path.join(output_dir, f"X_test{suffix}.csv"), index=False)
        final_val.to_csv(os.path.join(output_dir, f"X_val{suffix}.csv"), index=False)
        
        print(f"Final results:")
        print(f"  Training set: {len(final_train)} samples")
        print(f"  Validation set: {len(final_val)} samples")
        print(f"  Test set: {len(final_test)} samples")


def cross_validation_mode(df: pd.DataFrame,
                         n_clusters: int,
                         n_folds: int,
                         output_dir: str,
                         random_seed: int = 42) -> None:
    print(f"\n=== {n_folds}-Fold Cross-Validation Mode ===")
    print(f"Total dataset size: {len(df)} compounds")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of folds: {n_folds}")
    print(f"Minimum cluster size threshold: {n_folds}")
    
    fingerprints = calculate_ecfp4_fingerprints(df['SMILES'].tolist())
    cluster_labels = perform_clustering(fingerprints, n_clusters, random_state=42)
    df['cluster'] = cluster_labels
    min_cluster_size = n_folds
    df = merge_small_clusters_to_one(df, 'cluster', min_cluster_size)
    
    # 收集所有余数和合并族数据
    all_remainders = []
    unique_clusters = [c for c in df['cluster'].unique() if c != -1]
    
    print(f"\n--- Large Clusters Processing ---")
    print(f"Found {len(unique_clusters)} large clusters (size >= {min_cluster_size})")
    
    # 先分配满足阈值的大族
    cluster_fold_indices = {}
    total_large_cluster_samples = 0
    total_remainders = 0
    
    for cluster_id in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_id].copy()
        n = len(cluster_data)
        total_large_cluster_samples += n
        
        # 计算每个折应该分到的样本数
        base_size = n // n_folds
        remainder = n % n_folds
        
        print(f"\nCluster {cluster_id}: {n} compounds")
        print(f"  Base size per fold: {base_size}")
        print(f"  Remainder: {remainder}")
        
        # 先按基础大小分配
        fold_indices = []
        start_idx = 0
        for i in range(n_folds):
            end_idx = start_idx + base_size
            fold_indices.append(np.arange(start_idx, end_idx))
            start_idx = end_idx
        
        # 收集余数
        if remainder > 0:
            remainder_indices = np.arange(n - remainder, n)
            all_remainders.append(cluster_data.iloc[remainder_indices])
            total_remainders += remainder
            print(f"  Collected {remainder} remainder compounds")
        
        cluster_fold_indices[cluster_id] = fold_indices
    
    print(f"\nTotal large cluster samples: {total_large_cluster_samples}")
    print(f"Total remainders from large clusters: {total_remainders}")
    
    # 合并族数据
    merged_data = df[df['cluster'] == -1].copy()
    if len(merged_data) > 0:
        all_remainders.append(merged_data)
        print(f"\n--- Small Clusters Processing ---")
        print(f"Found 1 merged cluster (combining all small clusters): {len(merged_data)} compounds")
        print(f"  Added to remainder pool")
    
    # 将所有余数和合并族数据合并
    if all_remainders:
        combined_remainder_data = pd.concat(all_remainders, ignore_index=True)
        n_remainder = len(combined_remainder_data)
        print(f"\n--- Combined Remainder Processing ---")
        print(f"Combined remainder data: {n_remainder} samples")
        
        # 对合并的余数数据按比例分配
        base_size_remainder = n_remainder // n_folds
        final_remainder = n_remainder % n_folds
        
        print(f"  Base size per fold: {base_size_remainder}")
        print(f"  Final remainder: {final_remainder}")
        
        remainder_fold_indices = []
        start_idx = 0
        for i in range(n_folds):
            end_idx = start_idx + base_size_remainder
            remainder_fold_indices.append(np.arange(start_idx, end_idx))
            start_idx = end_idx
        
        # 最终余数随机分配
        if final_remainder > 0:
            final_remainder_indices = np.arange(n_remainder - final_remainder, n_remainder)
            np.random.seed(random_seed)
            random_folds = np.random.choice(n_folds, final_remainder, replace=False)
            print(f"  Final {final_remainder} samples randomly assigned to folds: {[f+1 for f in random_folds]}")
            for i, fold_idx in enumerate(random_folds):
                remainder_fold_indices[fold_idx] = np.append(remainder_fold_indices[fold_idx], 
                                                           final_remainder_indices[i])
    else:
        remainder_fold_indices = [[] for _ in range(n_folds)]
        print(f"\nNo remainder data to process")
    
    print(f"\n--- Final Fold Generation ---")
    
    # 生成各折的训练和验证集
    for fold in range(n_folds):
        train_dfs, val_dfs = [], []
        
        # 处理大族数据
        for cluster_id in unique_clusters:
            cluster_data = df[df['cluster'] == cluster_id].copy()
            val_idx = cluster_fold_indices[cluster_id][fold]
            train_idx = np.concatenate([cluster_fold_indices[cluster_id][i] for i in range(n_folds) if i != fold])
            val_dfs.append(cluster_data.iloc[val_idx])
            train_dfs.append(cluster_data.iloc[train_idx])
        
        # 处理余数数据
        if all_remainders:
            val_idx = remainder_fold_indices[fold]
            train_idx = np.concatenate([remainder_fold_indices[i] for i in range(n_folds) if i != fold])
            val_dfs.append(combined_remainder_data.iloc[val_idx])
            train_dfs.append(combined_remainder_data.iloc[train_idx])
        
        final_train = pd.concat(train_dfs, ignore_index=True)
        final_val = pd.concat(val_dfs, ignore_index=True)
        
        final_train.reset_index(drop=True, inplace=True)
        final_val.reset_index(drop=True, inplace=True)
        
        final_train.to_csv(os.path.join(output_dir, f"X_train{fold + 1}.csv"), index=False)
        final_val.to_csv(os.path.join(output_dir, f"X_val{fold + 1}.csv"), index=False)
        
        print(f"Fold {fold + 1}: Training set: {len(final_train)} samples, Validation set: {len(final_val)} samples")


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