import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, TransformerConv
from torch_geometric.nn import BatchNorm

class CrystalGNN(nn.Module):
    """
    Inputs:
        num_features (int): Number of input features per node
        hidden_channels (int, optional): Number of hidden channels. Defaults to 128
    """
    def __init__(self, num_features, hidden_channels=128, global_feature_size=2):
        super(CrystalGNN, self).__init__()

        #3 Convolutional layers with batch normalization
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.attn = TransformerConv(hidden_channels, hidden_channels, heads=2, concat=False)


        self.fc1 = nn.Linear(hidden_channels*2, hidden_channels)  # 256 -> 128
        self.fc2 = nn.Linear(hidden_channels, hidden_channels//2)  # 128 -> 64
        self.fc3 = nn.Linear(hidden_channels//2, 2)   # 64 -> 2 (Final classification)

        self.dropout = nn.Dropout(0.2)

        
        # Additional layers for global features
        self.global_processor = torch.nn.Linear(global_feature_size, hidden_channels)
        
        # Final classifier
        self.classifier = torch.nn.Linear(hidden_channels * 2, 2)  # *2 for concatenation

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x)

        #Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x)

        #Third layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        #transformer layer
        x = self.attn(x, edge_index)
        x = F.relu(x)

        #Pooling layer
        x_mean = global_mean_pool(x, batch)
        gf = self.global_processor(data.global_features)
        x = torch.cat([x_mean, gf], dim=1)

        #Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
        
        
        
        
        
        

