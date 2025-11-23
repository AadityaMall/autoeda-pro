# tests/ingestion_test.py
from src.ingestion import load_file, save_RawData

df, meta = load_file("demo/titanic.csv")
print(df.head())
print(meta)

path = save_RawData(df)
print("Saved to:", path)