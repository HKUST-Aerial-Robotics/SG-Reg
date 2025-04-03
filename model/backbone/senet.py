import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.execitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Input, 
            - x: (N,C)
        '''
        n, c = x.size()
        y = self.squeeze(x.t()).t() # (C)
        y = self.execitation(y) # (C,1)
        out = x * y.expand_as(x) # (N,C)
        return out
    

if __name__=='__main__':
    se = SEModule(32)
    x = torch.randn(4,32)
    out = se(x)
    print(out.shape)