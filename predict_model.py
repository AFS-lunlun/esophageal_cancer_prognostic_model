import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ================= Prediction Script: Load saved model and predict on new data =================
def predict_on_new_data(input_excel_path, model_pkl_path, output_dir='prediction_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model components...")
    model_components = joblib.load(model_pkl_path)
    
    best_features = model_components['best_features']
    pure_numeric_features = model_components['pure_numeric_features']
    categorical_features = model_components['categorical_features']
    manual_mappings = model_components['manual_mappings']
    imputer_numeric = model_components['imputer_numeric']
    imputer_categorical = model_components['imputer_categorical']
    cox_model = model_components['cox_model']
    
    print(f"Model loaded successfully! Using {len(best_features)} features:")
    print(best_features)
    
    # 1. Load new data
    print(f"\nLoading new data from: {input_excel_path}")
    new_data = pd.read_excel(input_excel_path)
    print(f"Original rows: {len(new_data)}")
    
    # 2. Check required features
    missing_cols = [col for col in best_features if col not in new_data.columns]
    if missing_cols:
        raise ValueError(
            f"Error: Missing required features:\n{missing_cols}\n"
            "Ensure input Excel uses exact English column names."
        )
    
    # Keep required columns
    required_cols = best_features + ['OS_Time', 'Event']
    new_data = new_data[[col for col in required_cols if col in new_data.columns]]
    
    # 3. Preprocessing
    print("\nStarting preprocessing...")
    
    missing_values = ['NA', '/', 'unknown', '？', 'nan', 'NaN', '']
    new_data = new_data.replace(missing_values, np.nan)
    
    numeric_in_model = [col for col in pure_numeric_features if col in new_data.columns]
    categorical_in_model = [col for col in categorical_features if col in new_data.columns]
    
    for col in numeric_in_model:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    
    for col in numeric_in_model:
        if new_data[col].isna().any():
            new_data[[col]] = imputer_numeric.transform(new_data[[col]])
    
    for col in categorical_in_model:
        if new_data[col].isna().any():
            new_data[[col]] = imputer_categorical.transform(new_data[[col]])
    
    for col, mapping in manual_mappings.items():
        if col in new_data.columns:
            new_data[col] = new_data[col].map(mapping).fillna(-1)
    
    if 'ECOG_Score' in new_data.columns:
        new_data['ECOG_Score'] = pd.to_numeric(new_data['ECOG_Score'], errors='coerce').fillna(-1)
    
    print("Skipping standardization (not required for Cox prediction; grouping remains consistent)")
    print("Preprocessing completed!")
    
    # 4. Predict Risk Score
    print("\nPredicting Risk Score using trained Cox model...")
    risk_scores = cox_model.predict_partial_hazard(new_data[best_features])
    new_data['Risk_Score'] = risk_scores.values.flatten()
    
    # 5. Risk grouping
    unique_scores = new_data['Risk_Score'].nunique()
    if unique_scores >= 3:
        new_data['Risk_Group'] = pd.qcut(new_data['Risk_Score'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    elif unique_scores == 2:
        new_data['Risk_Group'] = pd.qcut(new_data['Risk_Score'], q=2, labels=['Low', 'High'], duplicates='drop')
    else:
        new_data['Risk_Group'] = 'Medium'
    
    # 6. Optional: Calculate C-index if Event and OS_Time present
    if 'Event' in new_data.columns and 'OS_Time' in new_data.columns:
        print("\nCalculating C-index on provided data (for validation)...")
        c_index = concordance_index(new_data['OS_Time'], -new_data['Risk_Score'], new_data['Event'])
        print(f"C-index on this dataset: {c_index:.4f}")
    else:
        print("\nNote: Event/OS_Time not present - skipping C-index calculation (normal for new patients)")
    
    # 7. Save results
    output_csv = os.path.join(output_dir, 'predictions_with_risk.csv')
    new_data.to_csv(output_csv, index=False)
    print(f"\nPrediction completed!")
    print(f"Final samples: {len(new_data)}")
    print("Risk_Group distribution:")
    print(new_data['Risk_Group'].value_counts())
    print(f"Results saved to: {output_csv}")
    
    return new_data

# ================= Command-line execution =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esophageal Cancer Prognostic Model - Prediction Script")
    parser.add_argument('--input', '-i', required=True, help='Path to new patient data Excel file')
    parser.add_argument('--model', '-m', default='final_result/trained_model_components.pkl', help='Path to trained model pickle file')
    parser.add_argument('--output', '-o', default='prediction_results', help='Output directory')
    
    args = parser.parse_args()
    
    predict_on_new_data(args.input, args.model, args.output)
