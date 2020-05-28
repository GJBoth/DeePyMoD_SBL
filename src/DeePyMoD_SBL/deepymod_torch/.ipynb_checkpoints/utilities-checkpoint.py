from itertools import product, combinations
import numpy as np
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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0, initial_epoch=1000, sparsity_update_period=500):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.initial_epoch = initial_epoch
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
        self.first_sparsity_epoch = 1e8
        self.sparsity_update_period = sparsity_update_period
    
    def __call__(self, epoch, val_loss, model, optimizer):
        if (epoch >= self.initial_epoch) and (self.first_sparsity_epoch == 1e8): # first part
            self.update_score(val_loss, model, optimizer, epoch)
        elif (epoch > self.first_sparsity_epoch) and ((epoch - self.first_sparsity_epoch) % self.sparsity_update_period == 0): # sparsity update
            self.early_stop = True
        else: # before initial epoch
            pass
            
    def update_score(self, val_loss, model, optimizer, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.first_sparsity_epoch = epoch
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'model_checkpoint.pt')
        torch.save(optimizer.state_dict(), 'optimizer_checkpoint.pt')
        self.val_loss_min = val_loss
        
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
        


class EarlyStop:
    def __init__(self, patience=50, ini_epoch=1000, minimal_update=0.0):
        # internal state params
        self.l1_min = None #minimum l1_norm so far
        
        self.masks_similar = False
        self.l1_previous_mask = 0
        
        self.epochs_since_improvement = 0 
        self.n_sparsity_applied = 0
        
        
        # convergence decision params
        self.patience = patience # number of epochs to wait for improvement
        self.initial_epoch = ini_epoch # after which iteration to track convergence
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
    
    def sparsity_converged(self, l1_norm, tol=torch.tensor(0.05)):
        # converged if sparsity mask the same as last time and l1 norm as well.
        if (self.masks_similar == True) and ((torch.abs(l1_norm - self.l1_previous_mask) < tol).item()):
            converged = True
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