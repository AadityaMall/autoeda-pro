from src.ingestion import load_file
from src.validation import validate_dataset

df, meta = load_file("demo/titanic.csv")
report = validate_dataset(df)

print(report)