# AutoEDA Pro

AutoEDA Pro is an automated data exploration, validation, feature suggestion, and baseline modeling tool built with Streamlit.

## Project Structure
data_raw/         - original uploaded datasets
data_processed/   - cleaned & transformed datasets
src/              - main Python source code
src/app_pages/    - Streamlit UI pages
reports/          - generated plots + HTML/PDF reports
models/           - saved baseline models + pipelines
tests/            - pytest unit tests
demo/             - sample datasets for testing
.streamlit/       - streamlit config

### Data Ingestion & Validation
- Load CSV/XLSX files
- Structural checks (empty file, duplicate columns)
- Type inference and mixed-type detection
- Missing value analysis
- Duplicate row detection
- Invalid value checks (negative ages, corrupt dates)
- Outputs validation_report.json