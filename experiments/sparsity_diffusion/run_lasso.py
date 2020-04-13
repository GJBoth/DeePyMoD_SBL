# Imports and settings
import numpy as np
from sklearn.linear_model import Lasso

np.random.seed(42)

# Loading data
data_dict = np.load('data.npy', allow_pickle=True).item()
time_deriv = data_dict['time_deriv']
theta = data_dict['theta']

# Initializing
noise_levels = np.linspace(1e-3, 1.0, 100)
result = []
repeats = 10

# Running Lasso
for noise in noise_levels:
    single_noise_result = []
    for repeat in np.arange(repeats):
        reg = Lasso(alpha=1e-5, fit_intercept=False)
        t = time_deriv + np.random.normal(scale=noise * np.std(time_deriv), size=time_deriv.shape)
        t_normed = t / np.linalg.norm(t)
        theta_normed = theta / np.linalg.norm(theta, axis=0, keepdims=True)
        reg.fit(theta_normed, t_normed)
        single_noise_result.append(reg.coef_)
    result.append(single_noise_result)

result = np.array(result)

# Thresholding and averaging
threshold = 0.01
thresholded_result = result
thresholded_result[np.abs(result) <= threshold] = 0.0
thresholded_result[np.abs(result) > threshold] = 1.0

correct_vec = np.zeros(12)
correct_vec[2] = 1.0

correct_result = np.all(thresholded_result == correct_vec[None, None, :], axis=2)

print(np.mean(correct_result, axis=1))

                    thresholded_result[-1, 0, :]

data_dict = {'noise_levels': noise_levels, 'fit': result}
