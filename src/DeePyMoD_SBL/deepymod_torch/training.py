import torch
import time

from .output import Tensorboard, progress
from .losses import reg_loss, mse_loss, l1_loss
from .sparsity import scaling, threshold

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
    

def train_dynamic(model, data, target, optimizer, max_iterations, loss_func_args={'sparsity_update_period': 200, 'start_sparsity_update': 5000}):
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
        loss = torch.sum(loss_reg) + torch.sum(loss_mse)
        
        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), 0.0)
            # Before writing to tensorboard, we need to fill the missing values with 0
            coeff_vectors_padded = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.squeeze()) for mask, coeff_vector in zip(model.constraints.sparsity_mask, coeff_vector_list)]
            scaled_coeff_vectors_padded = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.squeeze()) for mask, coeff_vector in zip(model.constraints.sparsity_mask, coeff_vector_scaled_list)]
            
            board.write(iteration, loss, loss_mse, loss_reg, loss_reg, coeff_vectors_padded, scaled_coeff_vectors_padded)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Updating sparsity pattern
        if (iteration >= loss_func_args['start_sparsity_update']) and (iteration % loss_func_args['sparsity_update_period'] == 0):
            with torch.no_grad():
                model.constraints.sparsity_mask = model.calculate_sparsity_mask(theta, time_deriv_list)
                
    board.close()


def train_logprob(model, data, target, optimizer, max_iterations, loss_func_args={'l1':1e-5}):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)
    
    sigma = loss_func_args['noise'] # noise parameter
    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_ll = torch.mean((prediction - target)**2, dim=0) * 1 / (2 * sigma**2) + 1/2 * torch.log(2 * pi * sigma**2)
        loss_mse = mse_loss(prediction, target)
        loss_l1 = l1_loss(coeff_vector_scaled_list, loss_func_args['l1'])
        loss = torch.sum(loss_reg) + torch.sum(loss_ll) + torch.sum(loss_l1)
        
        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_l1).item())
            board.write(iteration, loss, loss_mse, loss_reg, loss_ll, coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()
    
    
def train_dynamic_logprob(model, data, target, optimizer, max_iterations, loss_func_args={'sparsity_update_period': 200, 'start_sparsity_update': 5000}):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)
    
    sigma = loss_func_args['noise'] # noise parameter
    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       LL |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_ll = torch.mean((prediction - target)**2, dim=0) * 1 / (2 * sigma**2) + 1/2 * torch.log(2 * pi * sigma**2)
        loss_mse = mse_loss(prediction, target)
        loss = torch.sum(loss_reg) + torch.sum(loss_ll)
        
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
    
def train_dynamic_logprob_scaled(model, data, target, optimizer, max_iterations, loss_func_args={'sparsity_update_period': 200, 'start_sparsity_update': 5000}):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)
    
    sigma = loss_func_args['noise'] # noise parameter
    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       LL |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list, theta = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_mse = mse_loss(prediction, target)
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_ll_fit =  loss_reg * 1 / (2 * sigma**2) + 1/2 * torch.log(2 * pi * sigma**2)
        loss_ll = loss_mse * 1 / (2 * sigma**2) + 1/2 * torch.log(2 * pi * sigma**2)
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