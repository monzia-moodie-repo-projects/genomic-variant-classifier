# Genomic Variant Classifier
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
An ensemble machine learning system for classifying pathogenic genetic variants
using data from ClinVar, gnomAD, and UniProt databases.
## Project Overview
# Genomic Variant Pathogenicity Classifier

Python-based ensemble ML system for classifying genetic variants using ClinVar, gnomAD, and UniProt.

## Scope
- Integrates 3 biomedical databases
- ACMG-aligned feature engineering
- Ensemble of 3 models: Gradient Boosting, Random Forest, Neural Network
- Clinical validation metrics
- ~10K-100K variant dataset (not billions)

## Technical Stack
- Python 3.10+
- Pandas for data processing
- Scikit-learn, XGBoost, TensorFlow
- ClinVar API integration
## Quick Start
```bash
git clone https://github.com/monzia-moodie/genomic-variant-classifier.git
cd genomic-variant-classifier

```
## Author
**Monzia Moodie** - [@monzia-moodie](https://github.com/monzia-moodie)
## License
MIT License
