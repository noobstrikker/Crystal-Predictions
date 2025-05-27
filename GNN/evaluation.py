import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns



def evaluate_predictions(model, test_loader, device):
    """
    Get predictions and true labels from the model
    
    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        
    Returns:
        tuple: (y_true, y_pred, y_probs) where all are numpy arrays
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # Get probabilities
            probs = torch.sigmoid(output)
            
            # Convert logits to binary predictions
            pred = (probs > 0.5).float()
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

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
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Non-metal', 'Metal'],
               yticklabels=['Non-metal', 'Metal'])
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def evaluate_model_performance(model, test_loader, device, property_name='Property', save_plots=True):
    """
    Evaluate model performance using various metrics
    
    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        property_name: Name of the property being predicted
        save_plots: Whether to save the evaluation plots
        
    Returns:
        dict: Dictionary containing various performance metrics
    """
    y_true, y_pred, y_probs = evaluate_predictions(model, test_loader, device)
    
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
    
    # Generate plots
    if save_plots:
        plot_predictions(y_true, y_pred, property_name=property_name, save_path=f'confusion_matrix_{property_name}.png')
        plot_roc_curve(y_true, y_probs, f'roc_curve_{property_name}.png')

    return metrics