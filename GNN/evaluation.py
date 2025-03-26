import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

"""
Maybe add some visualizations of the model. Supervisor suggested https://pymatgen.org/pymatgen.vis.html.
Maybe use pymatgen.vis for visualizing the crystal structures.

I think it requires a state dictionary of a trained NN to do.
"""

"""
Will be changed a bit, when a model is trained, and we actually have a state dictionary for the best model.
"""


def evaluate_predictions(model, test_loader, device):
    """
    Evaluate model predictions and return true vs predicted values
    """
    
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(output.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics

def plot_predictions(y_true, y_pred, property_name='Property', save_path=None):
    """
    Scatter plot of predicted vs true values
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel(f'True {property_name}')
    plt.ylabel(f'Predicted {property_name}')
    plt.title(f'True vs Predicted {property_name}')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_model_performance(model, test_loader, device, property_name='Property', save_plots=True):
    """
    Model evaluation function
    """
    # Get predictions
    y_true, y_pred = evaluate_predictions(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print(f"\nModel Performance Metrics for {property_name}:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    # Create plots
    if save_plots:
        plot_predictions(y_true, y_pred, property_name, f'predictions_{property_name}.png')
    
    return metrics
