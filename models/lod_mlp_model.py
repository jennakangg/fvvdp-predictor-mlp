import torch
import torch.nn as nn

class LowestLODPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(LowestLODPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) 
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
        x = x.to(torch.float32)

        x = self.layer1(x)
        x = self.bn1(x)  
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.bn2(x) 
        x = self.relu(x)

        x = self.dropout(x)  
        x = self.output_layer(x)
        
        return x

