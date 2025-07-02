import os
import sys
import pandas as pd

# Increase CSV read limit
import csv
csv.field_size_limit(sys.maxsize)

def list_columns(df: pd.DataFrame):
    print("Columns in CSV:", list(df.columns))

def load_dataframe(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Loaded {len(df)} rows")
    return df

# Function to calulate the number of spam and ham emails
def count_spam_ham(df: pd.DataFrame, label_column: str = 'Spam/Ham'):
    spam_count = df[df[label_column].str.strip().str.lower() == 'spam'].shape[0]
    ham_count = df[df[label_column].str.strip().str.lower() == 'ham'].shape[0]
    print(f"Spam emails: {spam_count}, Ham emails: {ham_count}")
    return spam_count, ham_count

def check_missing_or_invalid_labels(df: pd.DataFrame, label_column: str = 'Spam/Ham'):
    missing_label_rows = df[df[label_column].isnull() | ~df[label_column].isin(['spam', 'ham'])]
    if not missing_label_rows.empty:
        print(f"Found {len(missing_label_rows)} rows with missing or invalid labels:")
        print(missing_label_rows[[label_column]].head())
    else:
        print("All rows have valid 'spam' or 'ham' labels.")

    
