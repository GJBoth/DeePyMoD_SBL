# Imports
import numpy as np
import torch

from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersDelta
from DeePyMoD_SBL.deepymod_torch.library_functions import library_1D_in
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepModDynamic
from DeePyMoD_SBL.deepymod_torch.training import train_dynamic_logprob_scaled
from sklearn.linear_model import LassoLarsIC


import time
from DeePyMoD_SBL.deepymod_torch.output import Tensorboard, progress
from DeePyMoD_SBL.deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from DeePyMoD_SBL.deepymod_torch.sparsity import scaling, threshold
from numpy import pi


# Defining training function
def train_dynamic_logprob_scaled_double(model, data, target, optimizer, max_iterations, loss_func_args={'sparsity_update_period': 200, 'start_sparsity_update': 5000}):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)
    
    #sigma = loss_func_args['noise'] # noise parameter
    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       LL |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_mse = mse_loss(prediction, target)
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_ll_fit = torch.log(2 * pi * loss_reg)  #optimal sigma
        loss_ll = torch.log(2 * pi * loss_mse) #optimal sigma
            
        loss = torch.sum(loss_ll_fit) + torch.sum(loss_ll)
        
        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_ll).item())
            # Before writing to tensorboard, we need to fill the missing values with 0
            coeff_vectors_padded = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.squeeze()) for mask, coeff_vector in zip(model.constraints.sparsity_mask, coeff_vector_list)]
            scaled_coeff_vectors_padded = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.squeeze()) for mask, coeff_vector in zip(model.constraints.sparsity_mask, coeff_vector_scaled_list)]
            
            board.write(iteration, loss, loss_mse, loss_reg, loss_ll, coeff_vectors_padded, scaled_coeff_vectors_padded)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Updating sparsity pattern
        if (iteration >= loss_func_args['start_sparsity_update']) and (iteration % loss_func_args['sparsity_update_period'] == 0):
            with torch.no_grad():
                model.constraints.sparsity_mask = model.calculate_sparsity_mask(theta, time_deriv_list)
                
    board.close()
    
# Settings and parameters
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

v = 0.1
A = 1.0

# Making grid
x = np.linspace(-3, 4, 100)
t = np.linspace(0.5, 5.0, 50)
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')

# Making data
dataset = Dataset(BurgersDelta, v=v, A=A)
X_train, y_train = dataset.create_dataset(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), n_samples=2000, noise=0.1, random=True)

# Running deepmod
estimator = LassoLarsIC(fit_intercept=False)
config = {'n_in': 2, 'hidden_dims': [30, 30, 30, 30, 30], 'n_out': 1, 'library_function':library_1D_in, 'library_args':{'poly_order':2, 'diff_order': 2}, 'sparsity_estimator': estimator}
model = DeepModDynamic(**config)

optimizer = torch.optim.Adam(model.network_parameters(), betas=(0.99, 0.999), amsgrad=True)
train_dynamic_logprob_scaled_double(model, X_train, y_train, optimizer, 20000, loss_func_args={'start_sparsity_update': 50000, 'sparsity_update_period': 200})

torch.save(model.state_dict(), 'data/deepmod_lstsq_logprob_scaled_double_optimal.pt')