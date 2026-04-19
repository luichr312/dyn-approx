import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from NN_coordination import make_forward
from main import heat_exact_solution
import jax.numpy as jnp
import scipy

alpha = 0
quad = 10
quad_type = 'simpson'
lambda_damp = 0
data = loadmat(f"saved_sols/saved_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_N_4-15_Eps_0.1-0.01-0.001-0.0001_T_0.25-0.5-0.75-1.mat")
saved_params = data["params"]
num_eps = saved_params.shape[0]
num_N = saved_params.shape[1]
d = 2
nn_forward, param_count = make_forward(d, 6)

vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])
resolution_plot = 50
axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
grid_plot = jnp.meshgrid(*axes_plot)
xx_plot = jnp.stack([g.ravel() for g in grid_plot])


Ts = [0.25,0.5,0.75,1]
for i, T in enumerate(Ts):
    errors = np.zeros((num_eps, num_N))
    err_estimate = np.zeros((num_eps, num_N))
    for e in range(num_eps):
        for n in range(num_N):
            learned_sol = nn_forward(saved_params[e,n,:,i], xx_plot)
            errors[e,n] = 2 * jnp.pi / resolution_plot * jnp.linalg.norm(learned_sol - heat_exact_solution(xx_plot, T))
    scipy.io.savemat(
        f'error_mats/errors_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_N_4-15_Eps_0.1-0.01-0.001-0.0001_T_{T}.mat',
        {
            'errors': errors,
            'errors_est': err_estimate,
            'N_s': 2 ** np.arange(4, 15)
        })