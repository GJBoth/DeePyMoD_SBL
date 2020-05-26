import torch
import numpy as np
import time

from .output import Tensorboard, progress
from .losses import reg_loss, mse_loss, l1_loss
from .sparsity import scaling, threshold
from .utilities import EarlyStop

from numpy import pi

def train(model, data, target, optimizer, max_iterations, loss_func_args={'l1':1e-5}):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_mse = mse_loss(prediction, target)
        loss_l1 = l1_loss(coeff_vector_scaled_list, loss_func_args['l1'])
        loss = torch.sum(loss_reg) + torch.sum(loss_mse) + torch.sum(loss_l1)
        
        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_l1).item())
            board.write(iteration, loss, loss_mse, loss_reg, loss_l1, coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_mse(model, data, target, optimizer, max_iterations, loss_func_args={}):
    '''Trains the deepmod model only on the MSE. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 

        # Calculating loss
        loss_mse = mse_loss(prediction, target)
        loss = torch.sum(loss_mse)

        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), 0, 0)
            board.write(iteration, loss, loss_mse, [0], [0], coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_deepmod(model, data, target, optimizer, max_iterations, loss_func_args):
    '''Performs full deepmod cycle: trains model, thresholds and trains again for unbiased estimate. Updates model in-place.'''
    # Train first cycle and get prediction
    train(model, data, target, optimizer, max_iterations, loss_func_args)
    prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)

    # Threshold, set sparsity mask and coeff vector
    sparse_coeff_vector_list, sparsity_mask_list = threshold(coeff_vector_list, sparse_theta_list, time_deriv_list)
    model.fit.sparsity_mask = sparsity_mask_list
    model.fit.coeff_vector = torch.nn.ParameterList(sparse_coeff_vector_list)
   
    print()
    print(sparse_coeff_vector_list)
    print(sparsity_mask_list)

    #Resetting optimizer for different shapes, train without l1 
    optimizer.param_groups[0]['params'] = model.parameters()
    print() #empty line for correct printing
    train(model, data, target, optimizer, max_iterations, dict(loss_func_args, **{'l1': 0.0}))
    

def train_dynamic(model, data, target, optimizer, max_iterations=10000, stopper_kwargs={}, log_dir=None):
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms, log_dir) # initializing custom tb board
    
    early_stopper = EarlyStop(**stopper_kwargs) # initializing early stopper
    converged = False
    
    # Training
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating l1 norm
        l1_norm = torch.stack([torch.sum(torch.abs(coeff_vector)) for coeff_vector in coeff_vector_scaled_list])
        
        # Calculating loss
        loss_mse = mse_loss(prediction, target)
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss = torch.sum(2 * torch.log(2 * pi * loss_mse) + loss_reg / loss_mse)
        
        # Write progress to command line
        if iteration % 25 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(l1_norm).item())
            
        # Write to tensorboard (we pad the sparse vectors with zeros so they get written correctly)
        coeff_vectors_padded = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.squeeze()) for mask, coeff_vector in zip(model.constraints.sparsity_mask, coeff_vector_list)]
        scaled_coeff_vectors_padded = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.squeeze()) for mask, coeff_vector in zip(model.constraints.sparsity_mask, coeff_vector_scaled_list)]
        board.write(iteration, loss, loss_mse, loss_reg, l1_norm, coeff_vectors_padded, scaled_coeff_vectors_padded)
        
        # Updating sparsity and or convergence
        if early_stopper.coeffs_converged(iteration, torch.sum(l1_norm)): # sum when multiple outputs
            with torch.no_grad():
                new_masks = model.calculate_sparsity_mask(theta, time_deriv_list)
                masks_similar = np.all([torch.equal(new_mask, old_mask) for new_mask, old_mask in zip(new_masks, model.constraints.sparsity_mask)])
                early_stopper.masks_similar = masks_similar # if masks are similar, sparsity has converged
                print('Updating mask.')
                if early_stopper.sparsity_converged(torch.sum(l1_norm)):
                    converged = True
                early_stopper.l1_previous_mask = torch.sum(l1_norm)
                model.constraints.sparsity_mask = new_masks
                print(new_masks)
                
        # Stop running if sparsity converged
        if converged:
            print('Sparsity converged. Stopping training.')
            break
        
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    board.close()

    

        