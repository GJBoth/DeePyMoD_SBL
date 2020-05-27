# Imports
import numpy as np
import torch

from phimal_utilities.data import Dataset_2D
from phimal_utilities.data.diffusion import AdvectionDiffusionGaussian2D

from DeePyMoD_SBL.deepymod_torch.library_functions import library_2Din_1Dout
from DeePyMoD_SBL.deepymod_torch.DeepMod import DeepModDynamic
from DeePyMoD_SBL.deepymod_torch.training import train_dynamic
from DeePyMoD_SBL.deepymod_torch.estimators import Clustering
from sklearn.linear_model import LassoLarsIC

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making grid
dataset = Dataset_2D(AdvectionDiffusionGaussian2D, D=1.0, x0=[0.0, 0.0], sigma=0.5, v=[1.0, 1.0])
x = np.linspace(-4, 4, 100)
t = np.linspace(0.0, 2.0, 50)
x_grid, y_grid, t_grid = np.meshgrid(x, x, t, indexing='ij')

X = np.concatenate([x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1)

noise_range = np.arange(0.0, 1.01, 0.2)
n_runs = 5

for noise_level in noise_range:
    for run in np.arange(n_runs):
        X_train, y_train = dataset.create_dataset(X, t_grid.reshape(-1, 1), n_samples=5000, noise=noise_level, random=True, return_idx=False, random_state=run) # use the same dataset for every run; only diff is in the network
        estimator = Clustering(estimator=LassoLarsIC(fit_intercept=False))
        config = {'n_in': 3, 'hidden_dims': [30, 30, 30, 30, 30], 'n_out': 1, 'library_function':library_2Din_1Dout, 'library_args':{'poly_order': 1, 'diff_order': 2}, 'sparsity_estimator': estimator}
        model = DeepModDynamic(**config)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True)
        train_dynamic(model, X_train, y_train, optimizer, 25000, stopper_kwargs={'patience': 100, 'initial_epoch': 5000}, log_dir=f'runs/cluster_{noise_level:.2f}_run_{run}/')



