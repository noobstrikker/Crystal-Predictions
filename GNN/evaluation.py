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
    Get predictions and true labels from the model
    
    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        
    Returns:
        tuple: (y_true, y_pred) where both are numpy arrays
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # Convert logits to binary predictions
            pred = (output > 0).float()
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def calculate_metrics(y_true, y_pred):
    """
    Inputs:
        y_true: numpy array of true values (binary labels)
        y_pred: numpy array of predicted values (binary predictions)

    Returns:
        dict: Dictionary containing metrics:
            - 'accuracy': Classification accuracy
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def plot_predictions(y_true, y_pred, property_name='Property', save_path=None):
    """
    Inputs:
        y_true: numpy array of true values (binary labels)
        y_pred: numpy array of predicted values (binary predictions)
        property_name: String name of the property being plotted
        save_path: Optional path to save the plot. If None, plot is not saved

    Returns:
        None
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {property_name}')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_model_performance(model, test_loader, device):
    """
    Evaluate model performance using various metrics
    
    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary containing various performance metrics
    """
    y_true, y_pred = evaluate_predictions(model, test_loader, device)
    
    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-10)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_true == 1) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return metrics