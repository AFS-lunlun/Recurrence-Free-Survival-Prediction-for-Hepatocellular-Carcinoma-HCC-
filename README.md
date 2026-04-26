# Recurrence-Free Survival Prediction in HCC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Predict cancer recurrence using transcriptomic data with survival analysis models (RSF & DeepSurv).

## 📋 Table of Contents
- [Background](#background)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🎯 Background
Hepatocellular carcinoma (HCC) has high recurrence rates after initial treatment (up to 70% within 5 years). This project develops and validates machine learning models to predict **recurrence-free survival (RFS)** using patients' transcriptomic profiles, helping identify high-risk patients who may benefit from more aggressive surveillance or adjuvant therapy.

**Key clinical question**: Can gene expression patterns predict which patients will experience tumor recurrence after treatment?

## 📊 Data Requirements

### Input Files
1. **Clinical data** (`TCGA-LIHC_clinical_followup_nte_three_file_merage_outer.csv`):
   - `submitter_id`: Patient identifier
   - `new_tumor_event_after_initial_treatment`: Recurrence status (YES/NO)
   - `days_to_new_tumor_event_after_initial_treatment`: Time to recurrence (days)
   - `days_to_last_followup`: Censoring time for event-free patients

2. **Transcriptomic data** (`TCGA-LIHC_transcriptome_profilingonts_mRNA_count_annovar_format.csv`):
   - Gene expression counts (raw counts or normalized)
   - Patient IDs as row/column identifiers

3. **Selected features** (`tcga_selected_features.csv`):
   - List of prognostic gene signatures
   - Column: `feature_name`

### Data Preprocessing Pipeline
- **Event definition**: `TRUE` if new tumor event occurred, `FALSE` otherwise
- **Time calculation**: Days to recurrence if event occurred, otherwise days to last follow-up
- **Expression processing**: Log2(x+1) transformation + median imputation
- **Normalization**: RobustScaler (median-centered, IQR-scaled)

## 🔧 Installation

### Clone repository
```bash
git clone https://github.com/yourusername/HCC-recurrence-prediction.git
cd HCC-recurrence-prediction
