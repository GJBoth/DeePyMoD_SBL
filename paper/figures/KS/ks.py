import numpy as np
import torch
from scipy.io import loadmat

from DeePyMoD_SBL.deepymod_torch.library_functions import library_1D_in
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepModDynamic
from DeePyMoD_SBL.deepymod_torch.training import train_dynamic

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
# Prepping data
data = loadmat('kuramoto_sivishinky.mat')

t = data['tt']
x = data['x']
u = data['uu']
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')

x_grid = x_grid[:, :100]
t_grid = t_grid[:, :100]
u = uu[:, :100]

X = np.transpose((t_grid.flatten(), x_grid.flatten()))
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