import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class CrystalGNN(nn.Module):
    """
    Inputs:
        num_features (int): Number of input features per node
        hidden_channels (int, optional): Number of hidden channels. Defaults to 64
    """
    def __init__(self, num_features, hidden_channels=64):
        super(CrystalGNN, self).__init__()
        self.conv = GCNConv(num_features, hidden_channels)
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
        return F.log_softmax(x, dim=1)