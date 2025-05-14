import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, global_mean_pool, global_add_pool, TransformerConv
from torch_geometric.nn import BatchNorm

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
        self.conv1 = TAGConv(hidden_channels, hidden_channels, K=3)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = TAGConv(hidden_channels, hidden_channels, K=3)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = TAGConv(hidden_channels, hidden_channels, K=3)
        self.bn3 = BatchNorm(hidden_channels)

        # Final layers
        self.fc1 = nn.Linear(hidden_channels*3, hidden_channels)  # Combined node and lattice features
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