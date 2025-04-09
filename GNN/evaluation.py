import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            if isinstance(output, tuple):  # Handle multi-output models
                output = output[0]  # Take classification output only
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())
            y_probs.extend(torch.exp(output[:, 1]).cpu().numpy())
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else float('nan'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred, save_path):
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

def evaluate_model_performance(model, test_loader, device, property_name='is_metal', save_plots=True):
    """Complete model evaluation with metrics and plots"""
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # Store predictions
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())
            y_probs.extend(torch.exp(output[:, 1]).cpu().numpy())  # Probability of class 1
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else float('nan')
    }
    
    # Print results
    print(f"\nFinal Model Performance ({property_name}):")
    print(f"• Accuracy:  {metrics['accuracy']:.4f}")
    print(f"• Precision: {metrics['precision']:.4f}")
    print(f"• Recall:    {metrics['recall']:.4f}") 
    print(f"• F1 Score:  {metrics['f1']:.4f}")
    print(f"• ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Generate plots
    if save_plots:
        plot_confusion_matrix(y_true, y_pred, f'confusion_matrix_{property_name}.png')
        plot_roc_curve(y_true, y_probs, f'roc_curve_{property_name}.png')
    
    return metrics
    
    return metrics