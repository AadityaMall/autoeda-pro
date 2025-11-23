from src.ingestion import load_file
from src.eda import generate_missingness_reports, generate_distribution_reports

df, meta = load_file("demo/titanic.csv")
artifacts = generate_distribution_reports(df)
artifacts2 = generate_missingness_reports(df)
print("Distribution artifacts:")
print(artifacts)

print("Missingness artifacts:")
print(artifacts2)



# Check files exist
import os
for k, p in artifacts.items():
    assert os.path.exists(p), f"{p} missing"
print("Missingness artifacts created:", artifacts)