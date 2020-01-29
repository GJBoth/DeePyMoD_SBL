import torch
import torch.nn as nn


class outer_product(nn.Module):
    '''Pytorch style linear layer which also calculates the derivatives w.r.t input. Has been written to be a thin wrapper around the pytorch layer. '''
    def __init__(self):
        super().__init__()
        self.tril_indices = torch.triu_indices(row=5, col=5, offset=1)
        
    def forward(self, input):
        '''Calculates output'''
        z = torch.matmul(input[:, :, None], input[:, None, :])[:, self.tril_indices[0], self.tril_indices[1]]
        
        return z
