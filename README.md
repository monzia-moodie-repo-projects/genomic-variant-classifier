# Genomic Variant Classifier

An ensemble machine learning system for classifying pathogenic genetic variants using data from ClinVar, gnomAD, and UniProt databases. This project implements a production-ready pipeline for variant

## PROJECT OVERVIEW

This project implements a comprehensive pipeline for:
- Integrating variant data from multiple biomedical databases
genomic-variant-classifier/ ├── src/ │ ├── data/ # Data loading and preprocessing │
├── features/ # Feature engineering modules │ ├── models/ # Model architectures and
training │ ├── evaluation/ # Evaluation metrics and analysis │ └── utils/ # Utility
functions ├── notebooks/ # Jupyter notebooks for analysis ├── tests/ # Unit and
integration tests ├── configs/ # Configuration files ├── scripts/ # CLI scripts for
pipeline execution ├── data/ # Data directories (not tracked) ├── models/ # Saved
models (not tracked) └── docs/ # Documentation
## Installation
```bash
# Clone the repository
git clone https://github.com/monzia-moodie/genomic-variant-classifier.git
cd genomic-variant-classifier
# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Install in development mode
pip install -e .

## Installation
```bash
# Clone the repository
git clone https://github.com/monzia-moodie/genomic-variant-classifier.git
cd genomic-variant-classifier
# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Install in development mode
pip install -e .
Quick Start
from src.data.pipeline import VariantDataPipeline
from src.models.ensemble import EnsembleClassifier
# Initialize data pipeline
pipeline = VariantDataPipeline(config_path="configs/default.yaml")
# Load and preprocess data
X_train, X_test, y_train, y_test = pipeline.prepare_data()
# Train ensemble model
model = EnsembleClassifier()
model.fit(X_train, y_train)
# Evaluate
results = model.evaluate(X_test, y_test)
print(f"AUROC: {results['auroc']:.4f}")

Key Features
Data Integration
ClinVar variant annotations and clinical significance
gnomAD population allele frequencies
UniProt protein functional annotations
Computational predictor scores (CADD, REVEL, SpliceAI)
Feature Engineering
Population frequency features (PM2 criterion alignment)
Functional domain annotations (PM1 criterion)
Conservation scores (PP3/BP4 criteria)
Protein impact predictions
Model Architecture
Gradient Boosting (XGBoost/LightGBM)
Random Forest with calibration
Neural network with attention mechanism
Confidence-weighted ensemble aggregation
Evaluation Framework
Stratified cross-validation for imbalanced data
Precision-recall analysis
Calibration assessment
Clinical utility metrics
Configuration
Edit configs/default.yaml to customize:
data:
clinvar_path: "data/raw/clinvar_variant_summary.txt"
gnomad_path: "data/raw/gnomad.genomes.v4.vcf.gz"
model:
ensemble_weights: [0.4, 0.35, 0.25] # XGB, RF, NN
calibration: isotonic
training:
n_splits: 5
stratify: true
random_state: 42
License
MIT License - see LICENSE file for details.
Author
Monzia Moodie
GitHub: @monzia-moodie
Acknowledgments
ClinVar database (NCBI)
gnomAD consortium
UniProt consortium
ACMG/AMP variant classification guidelines 