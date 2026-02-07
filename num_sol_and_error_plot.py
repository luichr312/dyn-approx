import numpy as np
import matplotlib.pyplot as plt
from NN_coordination import make_forward_dirichlet_bc
from main import heat_exact_solution
import jax.numpy as jnp
import os, pickle
from scipy.io import loadmat
from matplotlib.colors import Normalize

alpha = 0.2
quad = 10
quad_type = 'simpson'
lambda_damp = 0
data = loadmat(f"saved_sols/saved_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_N_4-15_Eps_0.1-0.01-0.001-0.0001_T_0.25-0.5-0.75-1.mat")
params = data['params'][-1,-1

if os.path.exists('saved_params/params_heat_initial_hs6_3e-5.pickle'):
    with open('saved_params/params_heat_initial_hs6_3e-5.pickle', 'rb') as f:
        params0 = pickle.load(f)
else:
    raise ValueError('params_heat_initial_now.pickle does not exist')

d = 2
nn_forward, param_count = make_forward_dirichlet_bc(d, 6)

resolution_plot = 50
vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])
axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
grid_plot = jnp.meshgrid(*axes_plot)
xx_plot = jnp.stack([g.ravel() for g in grid_plot])


Ts = [0, 0.5, 1]
T_ind = [None, 1, 3]

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

fig, axes = plt.subplots(len(Ts), 2, figsize=(8, 4*len(Ts)), constrained_layout=True)
data = np.zeros((len(Ts),2,50,50))

for i, T in enumerate(Ts):
    p_i = params0
    if not i == 0:
        p_i = params[:,T_ind[i]]
    data[i,0] = nn_forward(p_i, xx_plot).reshape(50,50)
    data[i,1] = np.abs(data[i,0]- heat_exact_solution(xx_plot, T).reshape(50,50))

norms = [Normalize(vmin=np.min(data[:,col]), vmax=np.max(data[:,col])) for col in range(2)]

for row in range(len(Ts)):
    titles = (rf"Numerical solution at time $T={Ts[row]}$", rf"Error at time $T={Ts[row]}$")
    for col in range(2):
        axes[row, col].pcolormesh(
            grid_plot[0], grid_plot[1],
            data[row][col],
            norm=norms[col],
            cmap="viridis",
            shading="auto"
        )
        axes[row, col].set_rasterized(True)
        axes[row, col].set_title(titles[col])
        axes[row, col].set_aspect("equal")


for col in range(2):
    fig.colorbar(axes[0, col].collections[0], ax=axes[:, col], orientation="horizontal",pad=0.02)


plt.savefig(f"plots/eps/numsol_vs_error_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_Eps_0.0001_N_{2**14}.eps", format="eps", dpi=300)
plt.show()