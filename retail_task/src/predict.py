#!/usr/bin/env python3
"""
Model Evaluation Script (predict.py)
------------------------------------
CLI for evaluating saved models using existing functions from bases.base.

Usage:
    python src/predict.py \
      --model_path models/final_model/ridge_model.pkl \
      --data_path data/Retail.csv \
      --metrics_output_path results/evaluation_metrics.txt \
      --predictions_output_path results/evaluation_predictions.csv
"""

import argparse
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to the path to import from bases
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from bases.base import RegressionModel, mse, rmse, r2_score
from retail_task.src.data_preprocessing import RetailModel


def mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def load_model(model_path):
    """Load a saved model from pickle file"""
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_metrics(metrics, output_path):
    """Save evaluation metrics to a text file"""
    with open(output_path, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
        f.write(f"R-squared (R²): {metrics['r2']:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")


def save_predictions(y_true, y_pred, output_path):
    """Save predictions to a CSV file"""
    results_df = pd.DataFrame({
        'true_values': y_true,
        'predictions': y_pred,
        'residuals': y_true - y_pred
    })
    results_df.to_csv(output_path, index=False)


def print_metrics(metrics):
    """Print metrics to console"""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Mean Squared Error (MSE):     {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error:      {metrics['rmse']:.4f}")
    print(f"R-squared (R²):               {metrics['r2']:.4f}")
    print(f"Mean Absolute Error (MAE):    {metrics['mae']:.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained regression model on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a model on training data
  python src/predict.py \\
    --model_path models/final_model/ridge_model.pkl \\
    --data_path data/Retail.csv \\
    --metrics_output_path results/evaluation_metrics.txt \\
    --predictions_output_path results/evaluation_predictions.csv
        """
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model file (.pkl)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the data CSV file that includes features and true labels')
    parser.add_argument('--metrics_output_path', type=str, required=True,
                       help='Path where evaluation metrics will be saved')
    parser.add_argument('--predictions_output_path', type=str, required=True,
                       help='Path where predictions will be saved')
    
    # Optional arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information during execution')
    
    args = parser.parse_args()
    
    try:
        model = load_model(args.model_path)
        
        retail_model = RetailModel()
        retail_model.df = pd.read_csv(args.data_path)
        retail_model.preprocess()
        X, y = retail_model.extract_x_y("total_sales")
        
        y_pred = model.predict(X)
        
        metrics = {
            'mse': mse(y, y_pred),
            'rmse': rmse(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mae': mae(y, y_pred)
        }
        
        print_metrics(metrics)
        save_metrics(metrics, args.metrics_output_path)
        save_predictions(y, y_pred, args.predictions_output_path)
        print(f"Evaluation completed successfully!")
        
    except FileNotFoundError:
        print(" Model not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f" Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
