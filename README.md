# Recurrence-Free Survival Prediction for Hepatocellular Carcinoma (HCC)

## Project Overview
This project implements and compares two machine learning approaches for predicting **recurrence-free survival (RFS)** in Hepatocellular Carcinoma (HCC) patients using TCGA-LIHC (Liver Hepatocellular Carcinoma) dataset. The models predict the probability of tumor recurrence after initial treatment based on transcriptomic expression profiles.

## Key Features
- **Recurrence-Free Survival (RFS) Analysis**: Focuses on time to first recurrence or progression-free interval (PFI)
- **Dual Model Comparison**: 
  - Random Survival Forest (RSF) - Traditional ensemble method
  - DeepSurv - Deep learning approach using Cox proportional hazards
- **Endpoints**: 
  - Primary: Time to recurrence (days)
  - Event: New tumor event after initial treatment (YES/NO)

## Data Information
- **Source**: TCGA-LIHC (Liver Hepatocellular Carcinoma)
- **Clinical Endpoint**: PFI (Progression-Free Interval) / Recurrence status
- **Features**: Selected transcriptomic expression profiles (genes)
- **Sample Size**: ~370 patients with complete follow-up data

## Methodology
1. **Preprocessing**: 
   - Log2(x+1) transformation for gene expression data
   - RobustScaler for feature normalization
   - Median imputation for missing values

2. **Model Architecture**:
   - **RSF**: 200 trees, max depth 5, min samples split 10
   - **DeepSurv**: 128→64→1 neurons with BatchNorm and Dropout (0.4/0.3)

3. **Evaluation Metrics**:
   - Concordance Index (C-index)
   - Log-rank test for risk group stratification
   - Kaplan-Meier survival curves

## Results
- RSF C-index: ~0.708
- DeepSurv C-index: [Trained model result]

## Key Findings
Comparison of traditional survival forests vs deep learning approaches for recurrence prediction in liver cancer.
