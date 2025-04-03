import torch
from data_retrival import *
from data_preprocessing import *
from graph_builder import *
from torch_geometric.data import DataLoader
from GNN.my_model import * 
from GNN.train import *




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
    OUT_CHANNELS = 32
    

    

    # Load dataset
    dataset = load_data_local("Mads5")  # small run first

    # We still need to convert the crystal objects into PyTorch Geometric Data objects. How will we represent the crystal structure?

    # Split dataset into train and test
    train_dataset,val_dataset,test_dataset = split_data(dataset)
    
    #train_dataset, test_dataset = train_test_split(
        #dataset, 
        #test_size=0.2, 
        #random_state=42
    #)
    
    graphed_data = build_graph_batch(extract_label(train_dataset))
    graphed_test_data = build_graph_batch(extract_label(test_dataset))

    #
    
    
    # Create data loaders
    train_loader = DataLoader(graphed_data, batch_size =BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(graphed_test_data, batch_size= BATCH_SIZE,)
    print(graphed_data)
    # Initialize model
    model = CrystalGNN(
        num_features = graphed_data[0].num_features,
        hidden_channels=HIDDEN_CHANNELS,
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

     # Evaluation of model performance (after training)
    metrics = evaluate_model_performance(model, test_loader, device, property_name='Target Property')
if __name__ == '__main__':
    main()