import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

FEATURES = [
    'explained_log_gdp_per_capita',
    'explained_social_support',
    'explained_healthy_life_expectancy',
    'explained_freedom',
    'explained_generosity',
    'explained_corruption',
]
TARGET = 'happiness_score'

def load_raw_data(path='data/raw/world_happiness_report_2005_2025.csv'):
    df = pd.read_csv(path)
    print(f'Loaded {len(df):,} rows | Columns: {df.columns.tolist()}')
    return df

def preprocess(df):
    df = df.copy()
    # Drop rows missing the target
    df = df.dropna(subset=[TARGET])
    # Fill missing features with column median
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f'Removed {before - len(df):,} duplicates')
    print(f'Final dataset: {len(df):,} rows')
    print(f'Years: {df["year"].min()} - {df["year"].max()}')
    print(f'Countries: {df["country"].nunique()}')
    return df.reset_index(drop=True)

def split_data(df, test_size=0.2, random_state=42):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')
    return X_train, X_test, y_train, y_test

def save_splits(df, out_dir='data/processed'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train.to_csv(f'{out_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{out_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{out_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{out_dir}/y_test.csv', index=False)
    df.to_csv(f'{out_dir}/clean.csv', index=False)
    print(f'Saved to {out_dir}/')

def load_splits(out_dir='data/processed'):
    X_train = pd.read_csv(f'{out_dir}/X_train.csv')
    X_test = pd.read_csv(f'{out_dir}/X_test.csv')
    y_train = pd.read_csv(f'{out_dir}/y_train.csv').squeeze()
    y_test = pd.read_csv(f'{out_dir}/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = load_raw_data()
    df = preprocess(df)
    save_splits(df)
    print('Done.')
