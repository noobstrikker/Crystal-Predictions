import torch
from data_retrival import *
from data_preprocessing import *
from graph_builder import *
from torch_geometric.data import DataLoader
from GNN.my_model import CrystalGNN
from GNN.train import train_model, evaluate_model, evaluate_model_performance




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
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_CHANNELS = 64
    OUT_CHANNELS = 2  # Binary classification (metal/non-metal)
    

    # Load dataset
    raw_data = load_data_local("Mads100", 100)  # small run first
    
    if not raw_data:
        raise ValueError("No data was loaded. Check your data file and path.")
    
    print(f"Loaded {len(raw_data)} crystal structures")
    
    # Convert crystal structures to PyG Data objects
    graph_data = []
    for crystal_obj, structure in raw_data:
        # Use is_metal as the label (1 for metal, 0 for non-metal)
        label = 1.0 if crystal_obj.is_metal else 0.0
        graph = build_graph(structure, label=label)
        graph_data.append(graph)
    
    if not graph_data:
        raise ValueError("Failed to convert any structures to graphs")
    
    print(f"Converted {len(graph_data)} structures to graphs")
    
    # Initialize model - use the first graph to determine num_features
    num_features = graph_data[0].x.shape[1]  # Get feature dimension from the first graph
    
    # Split dataset into train, validation and test
    train_dataset, val_dataset, test_dataset = split_data(graph_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = CrystalGNN(
        num_features=num_features,
        hidden_channels=HIDDEN_CHANNELS
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.NLLLoss()  # Use NLLLoss with log_softmax output
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Evaluate final model performance
    metrics = evaluate_model_performance(model, test_loader, device, property_name='is_metal')
    print(f"Test Performance: {metrics}")

if __name__ == '__main__':
    main()