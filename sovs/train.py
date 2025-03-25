import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import CrystalGNN
import numpy as np
from sklearn.model_selection import train_test_split
 
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
 
        # Forward pass
        output = model(batch)
        loss = criterion(output, batch.y.view(-1, 1))
 
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    return total_loss / len(train_loader)
 
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch.y.view(-1, 1))
            total_loss += loss.item()
 
    return total_loss / len(test_loader)
 
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_CHANNELS = 64
    OUT_CHANNELS = 32
 
    # Load your dataset here
    # dataset = YourDatasetClass()
 
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