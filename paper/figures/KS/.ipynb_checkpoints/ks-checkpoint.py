import numpy as np
import torch
from scipy.io import loadmat

from DeePyMoD_SBL.deepymod_torch.library_functions import library_1D_in
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepModDynamic
from DeePyMoD_SBL.deepymod_torch.training import train_dynamic
from DeePyMoD_SBL.deepymod_torch.estimators import PDEFIND
from sklearn.linear_model import LassoCV

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(42)
torch.manual_seed(42)


# Prepping data
data = loadmat('kuramoto_sivishinky.mat')

t = data['tt']
x = data['x']
u = data['uu']
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')

# Use non-chaotic part of KS
x_grid = x_grid[:, :50]
t_grid = t_grid[:, :50]
u = u[:, :50]

X = np.concatenate((t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)), axis=1)
y = u.reshape(-1, 1)

noise_level = 0.25
y_noisy = y + noise_level * np.std(y, axis=0) * np.random.randn(*y.shape)
number_of_samples = 25000

idx = np.random.permutation(y.shape[0])
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y_noisy[idx, :][:number_of_samples], dtype=torch.float32)

estimator = PDEFIND(lam=1e-3, dtol=0.1)
config = {'n_in': 2, 'hidden_dims': [20, 20, 20, 20, 20, 20, 20], 'n_out': 1, 'library_function':library_1D_in, 'library_args':{'poly_order':1, 'diff_order': 4}, 'sparsity_estimator': estimator}
model = DeepModDynamic(**config)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True)

train_dynamic(model, X_train, y_train, optimizer, 100000, stopper_kwargs={'initial_epoch':40000, 'patience': 1000}, log_dir=f'runs/')