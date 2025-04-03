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
        
        # Forward pass
        output = model(batch)
        
        # For binary classification, y should be a class index (0 or 1)
        # Convert from float to long and ensure it's a 1D tensor
        target = batch.y.long().view(-1)
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
            target = batch.y.long().view(-1)  # Convert to 1D tensor of long integers
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def main():

    """
    Inputs:
        None (uses hardcoded hyperparameters and loads data from files)
    
    Outputs:
        - Saves trained model to 'best_model.pth'
        - Prints training progress
        - Returns None
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluation of model performance (after training)
    metrics = evaluate_model_performance(model, test_loader, device, property_name='Target Property')
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_CHANNELS = 64
    OUT_CHANNELS = 32
    
    # Load dataset
    dataset = load_data_local("crystal_data", amount=1000)  # Dont have file name and amount yet...

    # We still need to convert the crystal objects into PyTorch Geometric Data objects. How will we represent the crystal structure?

    # Split dataset into train and test
    train_dataset, test_dataset = train_test_split(
        dataset, 
        test_size=0.2, 
        random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = CrystalGNN(
        num_features=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    best_test_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()