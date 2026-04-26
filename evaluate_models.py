"""
Model evaluation script for survival analysis
Reproduces results from trained RSF and DeepSurv models
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import warnings

import torch
import torch.nn as nn

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Arial',
    'axes.facecolor': 'white'
})

def build_deepsurv_model(input_dim):
    """
    Build DeepSurv model architecture.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    
    Returns
    -------
    torch.nn.Module
        DeepSurv neural network model
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1)
    )
    return model

def load_and_evaluate(save_dir="saved_models"):
    """
    Load pre-trained models and test data for evaluation.
    
    Parameters
    ----------
    save_dir : str
        Directory containing saved models and test data
    
    Returns
    -------
    tuple
        (test_results, rsf_c_index, ds_c_index)
    """
    logging.info(f"Loading test data and model weights from {save_dir}/...")
    
    # Check if directory exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory {save_dir} not found. Please run training script first.")
    
    # Load test data
    X_test_path = os.path.join(save_dir, 'X_test_scaled.csv')
    y_test_path = os.path.join(save_dir, 'y_test.csv')
    
    if not os.path.exists(X_test_path):
        raise FileNotFoundError(f"Test data not found: {X_test_path}")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Test labels not found: {y_test_path}")
    
    X_test_scaled = pd.read_csv(X_test_path)
    y_test_df = pd.read_csv(y_test_path)
    
    # Convert to structured array for scikit-survival
    y_test = np.array(list(zip(y_test_df['Status'], y_test_df['Time'])), 
                      dtype=[('Status', '?'), ('Time', '<f8')])
    
    # Evaluate RSF model
    logging.info("Loading and evaluating Random Survival Forest...")
    rsf_path = os.path.join(save_dir, 'rsf_model.pkl')
    if not os.path.exists(rsf_path):
        raise FileNotFoundError(f"RSF model not found: {rsf_path}")
    
    rsf_model = joblib.load(rsf_path)
    rsf_risk_scores = rsf_model.predict(X_test_scaled)
    rsf_c_index = concordance_index_censored(
        y_test['Status'], 
        y_test['Time'], 
        rsf_risk_scores
    )[0]
    logging.info(f"[Reproduced] RSF test C-index: {rsf_c_index:.3f}")
    
    # Evaluate DeepSurv model
    logging.info("Loading and evaluating PyTorch DeepSurv...")
    ds_path = os.path.join(save_dir, 'deepsurv_model.pth')
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"DeepSurv model not found: {ds_path}")
    
    input_dim = X_test_scaled.shape[1]
    ds_model = build_deepsurv_model(input_dim)
    
    # Load model weights
    # Note: If you saved with torch.save(model.state_dict()), use this:
    ds_model.load_state_dict(torch.load(ds_path, map_location='cpu'))
    
    # If you saved with torch.save(model), use this instead:
    # ds_model = torch.load(ds_path, map_location='cpu')
    
    ds_model.eval()
    
    with torch.no_grad():
        X_te_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
        ds_risk_scores = ds_model(X_te_tensor).squeeze().numpy()
    
    ds_c_index = concordance_index_censored(
        y_test['Status'], 
        y_test['Time'], 
        ds_risk_scores
    )[0]
    logging.info(f"[Reproduced] DeepSurv test C-index: {ds_c_index:.3f}")
    
    # Prepare results for plotting
    test_results = pd.DataFrame({
        'Time': y_test['Time'],
        'Event': y_test['Status'],
        'RSF_Risk_Score': rsf_risk_scores,
        'DeepSurv_Risk_Score': ds_risk_scores
    })
    
    return test_results, rsf_c_index, ds_c_index

def plot_kaplan_meier_comparison(test_results, rsf_c, ds_c, output_filename='km_comparison.pdf'):
    """
    Generate Kaplan-Meier survival curves for risk groups.
    
    Parameters
    ----------
    test_results : pd.DataFrame
        DataFrame containing time, event, and risk scores
    rsf_c : float
        C-index for RSF model
    ds_c : float
        C-index for DeepSurv model
    output_filename : str
        Output file name for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = [
        {'name': 'Random Survival Forest', 
         'score_col': 'RSF_Risk_Score', 
         'c_index': rsf_c, 
         'ax': axes[0]},
        {'name': 'DeepSurv (Neural Network)', 
         'score_col': 'DeepSurv_Risk_Score', 
         'c_index': ds_c, 
         'ax': axes[1]}
    ]
    
    for model_info in models:
        ax = model_info['ax']
        median_risk = test_results[model_info['score_col']].median()
        
        high_risk = test_results[test_results[model_info['score_col']] >= median_risk]
        low_risk = test_results[test_results[model_info['score_col']] < median_risk]
        
        # Log-rank test
        lr_test = logrank_test(
            high_risk['Time'], 
            low_risk['Time'], 
            event_observed_A=high_risk['Event'], 
            event_observed_B=low_risk['Event']
        )
        p_value = lr_test.p_value
        
        # Plot survival curves
        kmf = KaplanMeierFitter()
        
        kmf.fit(low_risk['Time'], 
                event_observed=low_risk['Event'], 
                label='Low Risk Group')
        kmf.plot_survival_function(ax=ax, color='#1f77b4', linewidth=2.5, ci_alpha=0.2)
        
        kmf.fit(high_risk['Time'], 
                event_observed=high_risk['Event'], 
                label='High Risk Group')
        kmf.plot_survival_function(ax=ax, color='#d62728', linewidth=2.5, ci_alpha=0.2)
        
        # Formatting
        ax.set_title(f"{model_info['name']}\n(PFI Analysis)", 
                    fontfamily='Arial', fontsize=14, pad=10)
        ax.set_xlabel('Time (Days)', fontfamily='Arial', fontsize=12)
        ax.set_ylabel('Recurrence-Free Probability', fontfamily='Arial', fontsize=12)
        
        ax.text(0.05, 0.05, 
                f"Log-rank P = {p_value:.2e}\nC-index = {model_info['c_index']:.3f}", 
                transform=ax.transAxes, fontsize=11, fontfamily='Arial', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Kaplan-Meier plot saved to: {output_filename}")

def main():
    """Main execution function."""
    try:
        # Evaluate models
        results_df, rsf_c, ds_c = load_and_evaluate(save_dir="saved_models")
        
        # Generate plots
        plot_kaplan_meier_comparison(results_df, rsf_c, ds_c)
        
        print("\n" + "="*50)
        print("Model reproduction completed successfully!")
        print(f"RSF C-index: {rsf_c:.3f}")
        print(f"DeepSurv C-index: {ds_c:.3f}")
        print("="*50)
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.error("Please ensure the training script has been run first.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()