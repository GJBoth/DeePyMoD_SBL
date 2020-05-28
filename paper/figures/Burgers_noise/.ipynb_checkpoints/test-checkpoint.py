# Imports
import numpy as np
import torch

from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersDelta

from DeePyMoD_SBL.deepymod_torch.library_functions import library_1D_in
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepModDynamic
from DeePyMoD_SBL.deepymod_torch.training import train_dynamic, train_dynamic_old
from DeePyMoD_SBL.deepymod_torch.estimators import Clustering, Threshold
from sklearn.linear_model import LassoLarsIC, LassoCV

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
dataset = Dataset(BurgersDelta, v=v, A=A)

noise_range = np.arange(0.0, 1.01, 0.05)#np.arange(0.0, 0.51, 0.05)
n_runs = 1

for noise_level in [0.3]:
    for run in np.arange(n_runs):
        X_train, y_train = dataset.create_dataset(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), n_samples=1000, noise=noise_level, random=True, return_idx=False, random_state=run) # use the same dataset for every run; only diff is in the network
        #estimator = Clustering(estimator=LassoLarsIC(fit_intercept=False))
        estimator = Threshold(0.1, LassoCV(cv=5, fit_intercept=False))
        config = {'n_in': 2, 'hidden_dims': [30, 30, 30, 30, 30], 'n_out': 1, 'library_function':library_1D_in, 'library_args':{'poly_order':2, 'diff_order': 3}, 'sparsity_estimator': estimator}
        model = DeepModDynamic(**config)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True)
        train_dynamic(model, X_train, y_train, optimizer, 6000, log_dir=f'testruns/old_hyperparams_low_noise/')

