from itertools import product, combinations
import sys
import torch

def string_matmul(list_1, list_2):
    ''' Matrix multiplication with strings.'''
    prod = [element[0] + element[1] for element in product(list_1, list_2)]
    return prod


def terms_definition(poly_list, deriv_list):
    ''' Calculates which terms are in the library.'''
    if len(poly_list) == 1:
        theta = string_matmul(poly_list[0], deriv_list[0]) # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
    else:
        theta_uv = list(chain.from_iterable([string_matmul(u, v) for u, v in combinations(poly_list, 2)]))  # calculate all unique combinations between polynomials
        theta_dudv = list(chain.from_iterable([string_matmul(du, dv)[1:] for du, dv in combinations(deriv_list, 2)])) # calculate all unique combinations of derivatives
        theta_udu = list(chain.from_iterable([string_matmul(u[1:], du[1:]) for u, du in product(poly_list, deriv_list)])) # calculate all unique combinations of derivatives
        theta = theta_uv + theta_dudv + theta_udu
    return theta

def create_deriv_data(X, max_order):
    '''
    Automatically creates data-deriv tuple to feed to derivative network. 
    Shape before network is (sample x order x input).
    Shape after network will be (sample x order x input x output).
    '''
    
    if max_order == 1:
        dX = (torch.eye(X.shape[1]) * torch.ones(X.shape[0])[:, None, None])[:, None, :]
    else: 
        dX = [torch.eye(X.shape[1]) * torch.ones(X.shape[0])[:, None, None]]
        dX.extend([torch.zeros_like(dX[0]) for order in range(max_order-1)])
        dX = torch.stack(dX, dim=1)
        
    return (X, dX)


class EarlyStop:
    def __init__(self, patience=100, ini_epoch=1000, max_rounds=5, minimal_update=1e-2):
        # internal state params
        self.l1_min = None #minimum l1_norm so far
        self.epochs_since_improvement = 0 
        self.n_sparsity_applied = 0
        
        # convergence decision params
        self.patience = patience # number of epochs to wait for improvement
        self.initial_epoch = ini_epoch # after which iteration to track convergence
        self.max_rounds = max_rounds # max number of times to apply sparsity
        self.minimal_update = minimal_update # minimum magnitude of update to count as update
    
    def coeffs_converged(self, iteration, l1_norm):
        '''Checks if coefficients are converged. If no decrease in L1 norm is seen for [patience] epochs, update.
        Patience grows with number of times sparsity is applied.'''
        # update internal state
        self.update_state(l1_norm)
        
        # check convergence
        if (self.epochs_since_improvement == self.patience * (1 + self.n_sparsity_applied)) and (iteration > self.initial_epoch): #increase patience every run
            converged = True
            self.epochs_since_improvement = 0
            self.n_sparsity_applied += 1
        else:
            converged = False
        
        return converged

    def sparsity_converged(self):
        '''Checks if entire model is converged. Convergence reached when [max_rounds] is reached.'''
        if self.n_sparsity_applied == self.max_rounds:
            converged = True
            print(f'Sparsity has been applied {self.n_sparsity_applied} times')
        else:
            converged = False
        return converged
    
    def update_state(self, l1_norm):
        '''Updates internal state used to decide convergence.'''
        if self.l1_min is None: #initialization
            self.l1_min = l1_norm
            self.epochs_since_improvement += 1
        elif self.l1_min - l1_norm > self.minimal_update: #update if better
            self.l1_min = l1_norm
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
