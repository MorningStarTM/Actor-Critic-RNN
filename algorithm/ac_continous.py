import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import os

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super(RNN, self).__init__()
        self.lr = lr
        self.hidden_size = hidden_size
        self.hidden_state_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(input_size + hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    
    def forward(self, input, hidden):
        x = torch.tensor(input, dtype=torch.float32)
        combined = torch.cat((x, hidden), 1)
        hidden = self.hidden_state_layer(combined)
        output = self.output_layer(combined)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    

