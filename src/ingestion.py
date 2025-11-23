import os
import pandas as pd
from datetime import datetime

def detect_file_type(filepath: str) -> str:
    filepath = filepath.lower()
    if filepath.endswith('.csv'):
        return 'csv'
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return 'excel'
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
def load_file(filepath: str):
    file_type = detect_file_type(filepath)
    if file_type == 'csv':
        df = pd.read_csv(filepath)
    elif file_type == 'excel':
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    metadata = get_basicMetadata(df)
    metadata['file_type'] = file_type
    return df, metadata


def save_RawData(df: pd.DataFrame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_data_{timestamp}.csv"
    save_path  = os.path.join("data_raw", filename)
    df.to_csv(save_path, index=False)
    return save_path

def get_basicMetadata(df):
    metadata = {}
    metadata['num_rows'] = df.shape[0]
    metadata['num_columns'] = df.shape[1]
    metadata['columns'] = df.columns.tolist()
    metadata['dtypes'] = df.dtypes.apply(lambda x: x.name).to_dict()
    metadata['missing_values'] = df.isnull().sum().to_dict()
    metadata['created_at'] = datetime.now().isoformat()
    return metadata