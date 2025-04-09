import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        # Verify batch has targets
        if batch.y is None:
            raise ValueError("Batch is missing target labels (batch.y is None)")
        
        batch = batch.to(device)
        optimizer.zero_grad()
        
        output = model(batch)
        loss = criterion(output, batch.y.squeeze().long())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_loss_model(model, test_loader, criterion, device):
    """Evaluate model on test data"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            if isinstance(output, tuple):  # Handle multi-output models
                output = output[0]  # Use classification output only
            
            loss = criterion(output, batch.y.squeeze().long())
            total_loss += loss.item()
    
    return total_loss / len(test_loader)