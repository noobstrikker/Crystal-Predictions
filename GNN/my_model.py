import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, TransformerConv
from torch_geometric.nn import BatchNorm
from torch.nn import LayerNorm

class CrystalGNN(nn.Module):
    """
    Inputs:
        num_features (int): Number of input features per node
        hidden_channels (int, optional): Number of hidden channels. Defaults to 128
    """
    def __init__(self, num_features, hidden_channels=128, global_feature_size=2):
        super(CrystalGNN, self).__init__()

        # Node feature processing
        self.node_encoder = nn.Linear(num_features, hidden_channels)
        
        # Lattice feature processing
        self.lattice_encoder = nn.Linear(9, hidden_channels)  # 9 lattice features
        
        # Graph convolution layers
        heads = 4
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=heads)
        self.bn1 = BatchNorm(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.bn2 = BatchNorm(hidden_channels * heads)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.bn3 = BatchNorm(hidden_channels * heads)

        # Final layers
        self.fc1 = nn.Linear((hidden_channels * heads * 2) + hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels//2)
        self.fc3 = nn.Linear(hidden_channels//2, 1)   # Single output for binary classification

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, batch):
        """
        Inputs:
            x (torch.Tensor): Node feature matrix [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges]
            batch: Node to graph assignment [num_nodes]
            lattice_features (torch.Tensor): Lattice features [num_graphs, 9]

        Returns:
            torch.Tensor: Logits for binary classification [batch_size]
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        lattice_features = batch.lattice_features

        # Process node features
        x = self.node_encoder(x)
        x = F.relu(x)

        # Process lattice features
        lattice_features = self.lattice_encoder(lattice_features)
        lattice_features = F.relu(lattice_features)

        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Pooling
        x_mean = global_mean_pool(x, batch_idx)
        x_sum = global_add_pool(x, batch_idx)
        
        # Combine node features with lattice features
        x = torch.cat([x_mean, x_sum, lattice_features], dim=1)

        # Final layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x.view(-1)  # Ensure output is [batch_size]
    

class CrystalGNNTransformer(nn.Module):
    """
    A GNN model that combines transformerconv layers with graph convolution layers.
    
    Inputs:
        num_features (int): Number of input features per node
        hidden_channels (int, optional): Number of hidden channels. Defaults to 128
        global_feature_size (int, optional): Size of global features. Defaults to 2
    """
    def __init__(self, num_features, hidden_channels=128, global_feature_size=2):
        super(CrystalGNNTransformer, self).__init__()

        # Node feature processing
        self.node_encoder = nn.Linear(num_features, hidden_channels)
        
        # Edge feature processing (4 features: distance + 3D direction vector)
        self.edge_encoder = nn.Linear(4, hidden_channels)
        
        # Lattice feature processing
        self.lattice_encoder = nn.Linear(9, hidden_channels)  # 9 lattice features
        
        # Transformer layers for node features with edge attributes
        self.transformer1 = TransformerConv(hidden_channels, hidden_channels, heads=4, beta=True, edge_dim=hidden_channels)
        self.transformer2 = TransformerConv(hidden_channels * 4, hidden_channels, heads=4, beta=True, edge_dim=hidden_channels)
        self.transformer3 = TransformerConv(hidden_channels * 4, hidden_channels, heads=4, beta=True, edge_dim=hidden_channels)
        
        # Batch normalization layers
        self.bn1 = BatchNorm(hidden_channels * 4)
        self.bn2 = BatchNorm(hidden_channels * 4)
        self.bn3 = BatchNorm(hidden_channels * 4)

        # Layer normalization layers
        self.ln1 = LayerNorm(hidden_channels * 4)
        self.ln2 = LayerNorm(hidden_channels * 4)
        self.ln3 = LayerNorm(hidden_channels * 4)
        
        # Graph attention layers for structural information with edge attributes
        self.conv1 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=hidden_channels)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, edge_dim=hidden_channels)
        
        # Final layers
        self.fc1 = nn.Linear(hidden_channels * 3, hidden_channels)  # Combined features
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, 1)  # Single output for binary classification
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, batch):
        """
        Forward pass of the model.
        
        Inputs:
            batch: PyG batch object containing:
                x (torch.Tensor): Node feature matrix [num_nodes, num_features]
                edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges]
                batch: Node to graph assignment [num_nodes]
                lattice_features (torch.Tensor): Lattice features [num_graphs, 9]
                
        Returns:
            torch.Tensor: Logits for binary classification [batch_size]
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        edge_attr = batch.edge_attr
        lattice_features = batch.lattice_features

        # Process node features
        x = self.node_encoder(x)
        x = F.relu(x)

        # Process edge features
        edge_attr = self.edge_encoder(edge_attr)
        edge_attr = F.relu(edge_attr)

        # Process lattice features
        lattice_features = self.lattice_encoder(lattice_features)
        lattice_features = F.relu(lattice_features)

        # Transformer layers with edge attributes
        x = self.transformer1(x, edge_index, edge_attr)
        x = self.ln1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)

        x = self.transformer2(x, edge_index, edge_attr)
        x = self.ln2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)

        x = self.transformer3(x, edge_index, edge_attr)
        x = self.ln3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Graph convolution layers with edge attributes
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Pooling
        x_mean = global_mean_pool(x, batch_idx)
        x_sum = global_add_pool(x, batch_idx)
        
        # Combine node features with lattice features
        x = torch.cat([x_mean, x_sum, lattice_features], dim=1)

        # Final layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x.view(-1)  # Ensure output is [batch_size]
