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
    Inputs:
        model: PyTorch GNN model to evaluate
        test_loader: PyTorch Geometric DataLoader containing test data
        device: torch.device for model computation

    Returns:
        tuple: (y_true, y_pred)
            - y_true: numpy array of true values (binary labels)
            - y_pred: numpy array of predicted values (binary predictions)
    """
    
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # Get the predicted class (0 or 1) from log-softmax output
            # By taking the argmax along dimension 1
            pred_class = output.argmax(dim=1)
            
            # Convert the target to the right format
            target = batch.y.squeeze().long()
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred_class.cpu().numpy())
    
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

def evaluate_model_performance(model, test_loader, device, property_name='Property', save_plots=True):
    """
    Inputs:
        model: PyTorch GNN model to evaluate
        test_loader: PyTorch Geometric DataLoader containing test data
        device: torch.device for model computation
        property_name: String name of the property being predicted
        save_plots: Boolean flag for saving prediction plots

    Returns:
        dict: Dictionary containing classification metrics
    """
    # Get predictions
    y_true, y_pred = evaluate_predictions(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print(f"\nModel Performance Metrics for {property_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Create plots
    if save_plots:
        plot_predictions(y_true, y_pred, property_name, f'confusion_matrix_{property_name}.png')
    
    return metrics