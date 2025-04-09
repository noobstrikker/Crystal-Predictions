import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class CrystalGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, global_feature_size=2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Additional layers for global features
        self.global_processor = torch.nn.Linear(global_feature_size, hidden_channels)
        
        # Final classifier
        self.classifier = torch.nn.Linear(hidden_channels * 2, 2)  # *2 for concatenation

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Process local structure
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        # Global pooling
        x_global = global_mean_pool(x, batch)
        
        # Process global features if they exist
        if hasattr(data, 'global_features') and data.global_features is not None:
            gf = self.global_processor(data.global_features)
            x_global = torch.cat([x_global, gf], dim=1)
        
        # Final classification
        return self.classifier(x_global)