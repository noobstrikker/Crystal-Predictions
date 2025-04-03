import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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
    Inputs:
        model: PyTorch GNN model to evaluate
        test_loader: PyTorch Geometric DataLoader containing test data
        device: torch.device for model computation

    Returns:
        tuple: (y_true, y_pred)
            - y_true: numpy array of true values
            - y_pred: numpy array of predicted values
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

def calculate_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate classification metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_scores: Probability scores for the positive class
        
    Returns:
        dict: Dictionary of performance metrics
    """
    metrics = {}
    
    # Check if we have more than one class in the true labels
    unique_classes = np.unique(y_true)
    
    # Basic metrics that work with a single class
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Only calculate these metrics if we have more than one class
    if len(unique_classes) > 1:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Add ROC AUC if scores are provided and we have more than one class
        if y_scores is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    else:
        # If only one class is present
        metrics['precision'] = 1.0 if np.all(y_pred == y_true) else 0.0
        metrics['recall'] = 1.0 if np.all(y_pred == y_true) else 0.0
        metrics['f1'] = 1.0 if np.all(y_pred == y_true) else 0.0
        metrics['roc_auc'] = 'undefined (only one class present)'
    
    # Add confusion matrix - handle the case with only one class
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Ensure the confusion matrix has the right shape
    if cm.size == 1:  # Only one class
        if unique_classes[0] == 0:  # Only class 0
            tn = cm[0, 0]
            fp, fn, tp = 0, 0, 0
        else:  # Only class 1
            tp = cm[0, 0]
            tn, fp, fn = 0, 0, 0
    else:  # Both classes
        tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # Add class distribution information
    metrics['class_distribution'] = {
        'class_0_count': np.sum(y_true == 0),
        'class_1_count': np.sum(y_true == 1)
    }
    
    return metrics

def plot_predictions(y_true, y_pred, property_name='Property', save_path=None):
    """
    Inputs:
        y_true: numpy array of true values
        y_pred: numpy array of predicted values
        property_name: String name of the property being plotted
        save_path: Optional path to save the plot. If None, plot is not saved

    Returns:
        None
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

def evaluate_model_performance(model, test_loader, device, property_name='is_metal'):
    """
    Evaluate model performance on test data with various metrics.
    
    Args:
        model: Trained GNN model
        test_loader: DataLoader with test data
        device: torch device
        property_name: Name of the property being predicted
        
    Returns:
        dict: Dictionary of performance metrics
    """
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # Get predicted class (0 or 1)
            _, pred = output.max(dim=1)
            
            # Store true labels and predictions
            y_true.extend(batch.y.long().view(-1).cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
            # Store probability scores for ROC-AUC
            probs = torch.exp(output)  # Convert log_softmax to probabilities
            y_scores.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    # Convert to numpy arrays
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Print class distribution
    print(f"Class distribution in test set: {np.bincount(y_true)}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    return metrics
