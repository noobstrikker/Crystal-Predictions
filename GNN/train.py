import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from GNN.my_model import *
import numpy as np
from sklearn.model_selection import train_test_split
from GNN.evaluation import evaluate_model_performance


def train_model(model, train_loader, optimizer, criterion, device):
    """
    Inputs:
        model: CrystalGNN - The neural network model to train
        train_loader: DataLoader - PyTorch geometric dataloader containing training data
        optimizer: torch.optim - The optimizer for updating model parameters
        criterion: torch.nn - The loss function
        device: torch.device - The device (CPU/GPU) to run training on
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        output = model(batch)
        # Ensure output and target are both float tensors with correct shapes
        output = output.float().view(-1)  # Shape: [batch_size]
        target = batch.y.float().view(-1)  # Shape: [batch_size]
        
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    """
    Inputs:
        model: CrystalGNN - The neural network model to evaluate
        test_loader: DataLoader - PyTorch geometric dataloader containing test data
        criterion: torch.nn - The loss function
        device: torch.device - The device (CPU/GPU) to run evaluation on
    
    Returns:
        float: Average test loss
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            # Ensure output and target are both float tensors with correct shapes
            output = output.float().view(-1)  # Shape: [batch_size]
            target = batch.y.float().view(-1)  # Shape: [batch_size]
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)
