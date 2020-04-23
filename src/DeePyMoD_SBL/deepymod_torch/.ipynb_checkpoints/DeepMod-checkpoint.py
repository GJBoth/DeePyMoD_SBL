import torch
import torch.nn as nn
from .network import Fitting, Library, FittingDynamic


class DeepMod(nn.Module):
    ''' Class based interface for deepmod.'''
    def __init__(self, n_in, hidden_dims, n_out, library_function, library_args):
        super().__init__()
        self.network = self.build_network(n_in, hidden_dims, n_out)
        self.library = Library(library_function, library_args)
        self.fit = self.build_fit_layer(n_in, n_out, library_function, library_args)

    def forward(self, input):
        prediction = self.network(input)
        time_deriv, theta = self.library((prediction, input))
        sparse_theta, coeff_vector = self.fit((theta, time_deriv))
        return prediction, time_deriv, sparse_theta, coeff_vector, theta

    def build_network(self, n_in, hidden_dims, n_out):
        # NN
        network = []
        hs = [n_in] + hidden_dims + [n_out]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            network.append(nn.Linear(h0, h1))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        network = nn.Sequential(*network) 

        return network

    def build_fit_layer(self, n_in, n_out, library_function, library_args):
        sample_input = torch.ones((1, n_in), dtype=torch.float32, requires_grad=True)
        n_terms = self.library((self.network(sample_input), sample_input))[1].shape[1] # do sample pass to infer shapes
        fit_layer = Fitting(n_terms, n_out)

        return fit_layer

    # Function below make life easier
    def network_parameters(self):
        return self.network.parameters()

    def coeff_vector(self):
        return self.fit.coeff_vector.parameters()


class DeepModDynamic(nn.Module):
    ''' Class based interface for deepmod.'''
    def __init__(self, n_in, hidden_dims, n_out, library_function, library_args, sparsity_estimator):
        super().__init__()
        self.network = self.initialize_network(n_in, hidden_dims, n_out)
        self.library = Library(library_function, library_args)
        self.constraints = self.initialize_constraints_layer(n_in, n_out, library_function, library_args)
        self.sparsity_estimator = sparsity_estimator

    def forward(self, input):
        prediction = self.network(input)
        time_deriv, theta = self.library((prediction, input))
        sparse_theta, coeff_vector = self.constraints((theta, time_deriv))
        return prediction, time_deriv, sparse_theta, coeff_vector, theta

    def initialize_network(self, n_in, hidden_dims, n_out):
        # NN
        network = []
        hs = [n_in] + hidden_dims + [n_out]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            network.append(nn.Linear(h0, h1))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        network = nn.Sequential(*network) 

        return network

    def initialize_constraints_layer(self, n_in, n_out, library_function, library_args):
        sample_input = torch.ones((1, n_in), dtype=torch.float32, requires_grad=True)
        n_terms = self.library((self.network(sample_input), sample_input))[1].shape[1] # do sample pass to infer shapes
        
        fit_layer = FittingDynamic(n_terms, n_out)
        return fit_layer
    
    def calculate_sparsity_mask(self, theta, time_derivs):
        ''' Determines group sparsity mask from given scikit learn estimator. Theta and time derivs are normalized in here as well. Make sure to intercept = False in the estimator since we have a constant term in the library.'''
        # Normalizing inputs
        time_derivs_normed = [(time_deriv / torch.norm(time_deriv, keepdim=True)).detach().cpu().numpy() for time_deriv in time_derivs] 
        theta_normed = (theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu().numpy()
        
        # Fitting and determining sparsity mask
        coeff_vectors = [self.sparsity_estimator.fit(theta_normed, time_deriv_normed.squeeze()).coef_ for time_deriv_normed in time_derivs_normed]
        sparsity_masks = [torch.tensor(coeff_vector != 0.0, dtype=torch.bool) for coeff_vector in coeff_vectors]
        
        return sparsity_masks
        
    # Function below make life easier
    def network_parameters(self):
        return self.network.parameters()

    def coeff_vector(self):
        return self.fit.coeff_vector.parameters()
