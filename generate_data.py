#!/usr/bin/env python3
"""
Data Generation Script for AML/CTF Machine Learning Models

This script generates synthetic financial transaction data for training and testing
machine learning models for Anti-Money Laundering (AML) and Counter-Terrorist
Financing (CTF) detection.

Usage:
    python generate_data.py --output data/transactions.csv --count 10000
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_aml_dataset(n_transactions=10000, normal_ratio=0.90, random_seed=42):
    """
    Generate synthetic financial transaction dataset with AML/CTF patterns.
    
    Parameters:
    -----------
    n_transactions : int
        Total number of transactions to generate
    normal_ratio : float
        Proportion of normal (non-suspicious) transactions (0-1)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Generated transaction dataset with features and labels
    """
    
    np.random.seed(random_seed)
    
    logger.info(f"Generating {n_transactions} transactions ({normal_ratio*100:.0f}% normal)")
    
    # Calculate split
    n_normal = int(n_transactions * normal_ratio)
    n_suspicious = n_transactions - n_normal
    
    # Define high-risk countries (FATF Grey List)
    high_risk_countries = ['NG', 'SY', 'IR', 'KP', 'CU', 'VE', 'YE', 'LB', 'MM']
    normal_countries = ['USA', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'SG', 'CH']
    
    # ============ NORMAL TRANSACTIONS ============
    logger.info("Generating normal transactions...")
    
    normal_data = {
        'transaction_id': np.arange(n_normal),
        'amount': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'sender_age': np.random.normal(45, 15, n_normal),
        'receiver_age': np.random.normal(45, 15, n_normal),
        'transaction_count_sender': np.random.poisson(20, n_normal),
        'transaction_count_receiver': np.random.poisson(20, n_normal),
        'days_since_account_opened_sender': np.random.exponential(365, n_normal),
        'days_since_account_opened_receiver': np.random.exponential(365, n_normal),
        'avg_transaction_amount_sender': np.random.lognormal(mean=6.5, sigma=2, size=n_normal),
        'avg_transaction_amount_receiver': np.random.lognormal(mean=6.5, sigma=2, size=n_normal),
        'hour_of_day': np.random.randint(8, 18, n_normal),
        'day_of_week': np.random.randint(0, 7, n_normal),
        'country_sender': np.random.choice(normal_countries, n_normal),
        'country_receiver': np.random.choice(normal_countries, n_normal),
        'suspicious_label': 0
    }
    
    # ============ SUSPICIOUS TRANSACTIONS ============
    logger.info("Generating suspicious transactions...")
    
    suspicious_data = {
        'transaction_id': np.arange(n_normal, n_transactions),
        'amount': np.random.lognormal(mean=9, sigma=1.5, size=n_suspicious),  # Larger amounts
        'sender_age': np.random.choice([25, 30, 35, 65, 70, 75], n_suspicious),  # Unusual ages
        'receiver_age': np.random.choice([25, 30, 35, 65, 70, 75], n_suspicious),
        'transaction_count_sender': np.random.poisson(50, n_suspicious),  # More frequent
        'transaction_count_receiver': np.random.poisson(50, n_suspicious),
        'days_since_account_opened_sender': np.random.exponential(50, n_suspicious),  # Newer accounts
        'days_since_account_opened_receiver': np.random.exponential(50, n_suspicious),
        'avg_transaction_amount_sender': np.random.lognormal(mean=8.5, sigma=1.5, size=n_suspicious),
        'avg_transaction_amount_receiver': np.random.lognormal(mean=8.5, sigma=1.5, size=n_suspicious),
        'hour_of_day': np.random.randint(0, 24, n_suspicious),  # Any hour
        'day_of_week': np.random.randint(0, 7, n_suspicious),
        'country_sender': np.random.choice(high_risk_countries, n_suspicious),  # High-risk countries
        'country_receiver': np.random.choice(high_risk_countries, n_suspicious),
        'suspicious_label': 1
    }
    
    # Combine datasets
    logger.info("Combining and shuffling data...")
    
    df = pd.DataFrame(normal_data)
    df_suspicious = pd.DataFrame(suspicious_data)
    df = pd.concat([df, df_suspicious], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Add temporal features
    logger.info("Adding temporal features...")
    
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    
    # Add derived features
    logger.info("Engineering features...")
    
    df['round_amount'] = ((df['amount'] % 1000) == 0).astype(int)
    df['cross_border'] = (df['country_sender'] != df['country_receiver']).astype(int)
    df['high_risk_country_sender'] = df['country_sender'].isin(high_risk_countries).astype(int)
    df['high_risk_country_receiver'] = df['country_receiver'].isin(high_risk_countries).astype(int)
    
    logger.info("Dataset generation complete!")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Suspicious transactions: {df['suspicious_label'].sum()} ({df['suspicious_label'].mean()*100:.1f}%)")
    logger.info(f"Features: {len(df.columns)}")
    
    return df


def validate_dataset(df):
    """
    Validate generated dataset for quality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    """
    
    logger.info("Validating dataset...")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing values")
    else:
        logger.info("✓ No missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows")
    else:
        logger.info("✓ No duplicate rows")
    
    # Check data types
    logger.info(f"✓ Data types verified: {df.dtypes.unique().tolist()}")
    
    # Check class balance
    class_dist = df['suspicious_label'].value_counts()
    logger.info(f"✓ Class distribution: {class_dist.to_dict()}")
    
    # Check value ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        logger.info(f"  {col}: [{min_val:.2f}, {max_val:.2f}]")
    
    logger.info("✓ Dataset validation complete")


def save_dataset(df, output_path):
    """
    Save dataset to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to save
    output_path : str
        Path to save CSV file
    """
    
    logger.info(f"Saving dataset to {output_path}...")
    
    # Create directory if needed
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    logger.info(f"✓ Dataset saved successfully ({file_size:.2f} MB)")


def generate_data_splits(df, output_dir='data', test_size=0.2, random_seed=42):
    """
    Generate train-test splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    output_dir : str
        Directory to save splits
    test_size : float
        Proportion for test set
    random_seed : int
        Random seed for reproducibility
    """
    
    from sklearn.model_selection import train_test_split
    
    logger.info("Creating train-test splits...")
    
    X = df.drop('suspicious_label', axis=1)
    y = df['suspicious_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Save splits
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    logger.info(f"✓ Training set: {len(X_train)} samples")
    logger.info(f"✓ Test set: {len(X_test)} samples")
    logger.info(f"✓ Splits saved to {output_dir}/")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic AML/CTF transaction data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/transactions.csv',
        help='Output file path (default: data/transactions.csv)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10000,
        help='Number of transactions to generate (default: 10000)'
    )
    parser.add_argument(
        '--normal-ratio',
        type=float,
        default=0.90,
        help='Proportion of normal transactions (default: 0.90)'
    )
    parser.add_argument(
        '--splits',
        action='store_true',
        help='Generate train-test splits'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("AML/CTF TRANSACTION DATA GENERATION")
    logger.info("="*80)
    
    # Generate dataset
    df = generate_aml_dataset(
        n_transactions=args.count,
        normal_ratio=args.normal_ratio,
        random_seed=args.seed
    )
    
    # Validate
    validate_dataset(df)
    
    # Save
    save_dataset(df, args.output)
    
    # Generate splits if requested
    if args.splits:
        output_dir = os.path.dirname(args.output) or 'data'
        generate_data_splits(df, output_dir=output_dir, random_seed=args.seed)
    
    logger.info("="*80)
    logger.info("Data generation completed successfully!")
    logger.info("="*80)


if __name__ == '__main__':
    import os
    main()
