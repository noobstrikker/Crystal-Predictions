import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm

class CrystalGNN(nn.Module):
    """
    Inputs:
        num_features (int): Number of input features per node
        hidden_channels (int, optional): Number of hidden channels. Defaults to 128
    """
    def __init__(self, num_features, hidden_channels=128):
        super(CrystalGNN, self).__init__()

        #3 Convolutional layers with batch normalization
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        #Connected layers
        self.fc = nn.Linear(hidden_channels, 2)  
        
    def forward(self, batch):
        """
        Inputs:
            x (torch.Tensor): Node feature matrix [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges]
                where each column is [source_node, target_node]
            batch: Node to graph assignment [num_nodes]

        Returns:
            torch.Tensor: Log-softmax probabilities for binary classification [batch_size, 2]
        """

        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Pooling layer
        x = global_mean_pool(x, batch_idx)
        
        # Classification
        x = self.fc(x)
        # Process global features if they exist
        if hasattr(data, 'global_features') and data.global_features is not None:
            gf = self.global_processor(data.global_features)
            x_global = torch.cat([x_global, gf], dim=1)
        
        # Final classification
        return self.classifier(x_global)