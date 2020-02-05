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

class Library(nn.Module):
    '''Abstract baseclass for library-as-layer. Child requires theta function (see library_functions). '''
    def __init__(self, input_dim, output_dim, diff_order):
        super().__init__()
        self.diff_order = diff_order
        self.total_terms = self.terms(input_dim, output_dim, self.diff_order)

    def forward(self, input):
        '''Calculates output.'''
        time_deriv_list, theta = self.theta(input)
        return input, time_deriv_list, theta

    def terms(self, input_dim, output_dim, max_order):
        '''Calculates the number of terms the library produces'''
        sample_data = (torch.ones((1, output_dim), dtype=torch.float32), torch.ones((1, max_order, input_dim, output_dim), dtype=torch.float32)) # we run a single forward pass on fake data to infer shapes
        total_terms = self.theta(sample_data)[1].shape[1]

        return total_terms