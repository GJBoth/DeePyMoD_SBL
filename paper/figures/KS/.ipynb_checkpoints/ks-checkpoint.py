import numpy as np
import pandas as pd
import torch
import time
from scipy.io import loadmat

from DeePyMoD_SBL.deepymod_torch.library_functions import library_1D_in
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepModDynamic
from DeePyMoD_SBL.deepymod_torch.training import train_dynamic_logprob_scaled
from sklearn.linear_model import LassoLarsIC, LarsCV
from DeePyMoD_SBL.deepymod_torch.output import Tensorboard, progress
from DeePyMoD_SBL.deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from DeePyMoD_SBL.deepymod_torch.sparsity import scaling, threshold
from numpy import pi

from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersDelta
from phimal_utilities.analysis import load_tensorboard

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
# Defining training function
def train(model, data, target, optimizer, max_iterations, loss_func_args, log_dir=None):
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms, log_dir)
    
    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       LL |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_mse = mse_loss(prediction, target)
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_ll = torch.log(2 * pi * loss_mse)
        loss_ll_fit = torch.log(2 * pi * loss_mse) + loss_reg / loss_mse
        loss = torch.sum(loss_ll) + torch.sum(loss_ll_fit)
        
        # Writing
        if iteration % 100 == 0:
            # Write progress to command line
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_ll).item())
            
            # Calculate error for theta
            
            # Write to tensorboard
            board.write(iteration, loss, loss_mse, loss_reg, loss_ll, coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    board.close()

data = loadmat('kuramoto_sivishinky.mat')

t = data['tt']
x=data['x']
uu= data['uu']
x_grid,t_grid = np.meshgrid(x,t,indexing='ij')

x_grid = x_grid[:,:100]
t_grid = t_grid[:,:100]
uu = uu[:,:100]
X = np.transpose((t_grid.flatten(),x_grid.flatten()))
y = uu.reshape((uu.size, 1))

noise_level = 0.0
y_noisy = y + noise_level * np.std(y) * np.random.randn(y[:,0].size, 1)
number_of_samples = 20000

idx = np.random.permutation(y.shape[0])
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y_noisy[idx, :][:number_of_samples], dtype=torch.float32)

estimator = LarsCV(fit_intercept=False)

config = {'n_in': 2, 'hidden_dims': [20, 20, 20, 20, 20,20, 20], 'n_out': 1, 'library_function':library_1D_in, 'library_args':{'poly_order': 1, 'diff_order': 4}, 'sparsity_estimator': estimator}
model = DeepModDynamic(**config)
optimizer = torch.optim.Adam(model.network_parameters(),betas=(0.99,0.99), amsgrad=True)

train(model, X_train, y_train, optimizer, 100000, loss_func_args={'start_sparsity_update': 3000, 'sparsity_update_period': 250})