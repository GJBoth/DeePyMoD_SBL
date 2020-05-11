# Imports
import numpy as np
import torch

from phimal_utilities.data import Dataset_2D
from phimal_utilities.data.diffusion import AdvectionDiffusionGaussian2D
from DeePyMoD_SBL.deepymod_torch.library_functions import library_2Din_1Dout
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepMod

import time
from DeePyMoD_SBL.deepymod_torch.output import Tensorboard, progress
from DeePyMoD_SBL.deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from DeePyMoD_SBL.deepymod_torch.sparsity import scaling, threshold
from numpy import pi


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
        loss = torch.sum(loss_mse) + torch.sum(loss_reg)
        
        # Writing
        if iteration % 100 == 0:
            # Write progress to command line
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_reg).item())
            
            lstsq_solution = torch.inverse(theta.T @ theta) @ theta.T @ time_deriv_list[0]
            
            # Calculate error for theta
            theta_true = loss_func_args['library']
            dt_true = loss_func_args['time_deriv']
            mae_library = torch.mean(torch.abs(theta - theta_true), dim=0)
            mae_dt = torch.mean(torch.abs(dt_true - time_deriv_list[0]), dim=0)
            
            # Write to tensorboard
            board.write(iteration, loss, loss_mse, loss_reg, loss_reg, coeff_vector_list, coeff_vector_scaled_list, lstsq_solution=lstsq_solution.squeeze(), mae_library=mae_library, mae_time_deriv=mae_dt)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    board.close()

# Settings and parameters
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = Dataset_2D(AdvectionDiffusionGaussian2D, D=2.0, x0=[0.0, 0.0], sigma=0.5, v=[0.1, 0.1])
x = np.linspace(-10, 10, 100)
t = np.linspace(0.0, 1.0, 25)
x_grid, y_grid, t_grid = np.meshgrid(x, x, t, indexing='ij')

X = np.concatenate([x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1)

config = {'n_in': 3, 'hidden_dims': [30, 30, 30, 30, 30], 'n_out': 1, 'library_function':library_2Din_1Dout, 'library_args':{'poly_order':2, 'diff_order': 2}}


X_train, y_train, rand_idx = dataset.create_dataset(X,t_grid.reshape(-1,1), n_samples=1000, noise=0.1, random=True, return_idx=True)
theta = dataset.library(X,t_grid.reshape(-1,1))[rand_idx, :]
dt = dataset.time_deriv(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))[rand_idx, :]
model = DeepMod(**config)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True)
train(model, X_train, y_train, optimizer, 20000, loss_func_args={'library':torch.tensor(theta) ,'time_deriv': torch.tensor(dt)}, log_dir = f'runs/deepmod_run_script')
