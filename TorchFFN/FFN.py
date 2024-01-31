import torch
import torch.nn as nn
from torch.nn import functional as F

class Classification_NN(nn.Module):
    def __init__(self, input_dim = 30, hidden_dim = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.FFN = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, 1, bias=True),
                                 nn.Sigmoid()
                                )
        self.BCELoss = nn.BCELoss()
        self.apply(self.__init__weights)
    
    # weight initialization for guaranteed convergence
    def __init__weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self,x,y):
        out = self.FFN(x)
        loss = self.BCELoss(out,y)
        return loss,out 