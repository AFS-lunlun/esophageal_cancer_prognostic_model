# esophageal_cancer_prognostic_model

This repository provides the complete code for a Cox proportional hazards-based prognostic model in esophageal cancer patients treated with immunotherapy and Radiotherapy (as described in our manuscript).

The model uses 11 key clinical and laboratory features to calculate a Risk Score and stratify patients into Low/Medium/High risk groups. It also identifies patients likely to benefit from radiotherapy combined with immunotherapy.

## Key Results (Internal Validation)
- **C-index**: 0.7014
- **Calibration**: Excellent (Brier scores: 0.016–0.031 across 6–24 months)
- **Clinical implication**: Approximately 68% of patients (Risk Score < 1.19) show significant benefit from radiotherapy + immunotherapy (STEPP and IPTW analyses)

## Requirements

Python 3.9+ and packages listed in `requirements.txt`.

Install dependencies:
```bash
pip install -r requirements.txt
```

command：
```bash
python predict.py --input path/to/new_data.xlsx --model final_result/trained_model_components.pkl --output my_predictions
```
