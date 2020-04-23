# Imports
import numpy as np
import torch

from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersDelta
from DeePyMoD_SBL.deepymod_torch.library_functions import library_1D_in
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepModDynamic
from DeePyMoD_SBL.deepymod_torch.training import train_dynamic_logprob
from sklearn.linear_model import LassoLarsIC

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
sigma = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam([{'params': model.network_parameters(), 'betas': (0.99, 0.999)}, {'params': sigma, 'betas': (0.99, 0.999)}], amsgrad=True)

train_dynamic_logprob(model, X_train, y_train, optimizer, 20000, loss_func_args={'start_sparsity_update': 50000, 'sparsity_update_period': 200, 'noise': sigma})

torch.save(model.state_dict(), 'data/deepmod_lstsq_logprob.pt')