import numpy as np
import matplotlib.pyplot as plt
from NN_coordination import make_forward
from main import heat_exact_solution
import jax.numpy as jnp
import os, pickle
from scipy.io import loadmat

alpha = 0
quad = 10
quad_type = 'simpson'
lambda_damp = 0
data = loadmat(f"saved_sols/saved_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_N_4-15_Eps_0.1-0.01-0.001-0.0001_T_0.25-0.5-0.75-1.mat")
params = data['params'][-1,-1]

if os.path.exists('saved_params/params_heat_initial_hs6_3e-5.pickle'):
    with open('saved_params/params_heat_initial_hs6_3e-5.pickle', 'rb') as f:
        params0 = pickle.load(f)
else:
    raise ValueError('params_heat_initial_now.pickle does not exist')

d = 2
nn_forward, param_count = make_forward(d, 6)

resolution_plot = 50
vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])
axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
grid_plot = jnp.meshgrid(*axes_plot)
xx_plot = jnp.stack([g.ravel() for g in grid_plot])

Ts = [0,0.25,0.5,0.75,1]

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

fig = plt.figure(figsize=(5*len(Ts),6))
axes  = [fig.add_subplot(1, 5, i+1, projection='3d') for i in range(len(Ts))]
for i, ax in enumerate(axes):
    p_i = params0
    if not i == 0:
        p_i = params[:, i - 1]
    im = ax.plot_surface(grid_plot[0], grid_plot[1], nn_forward(p_i, xx_plot).reshape(50, 50), cmap="viridis")
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_title(f"$T = {Ts[i]}$", fontsize=25)
    ax.set_zlim(-0.1,1)
    #ax.set_aspect("equal", adjustable="box")
#fig.colorbar(im, ax=axes, shrink=0.8)

fig.suptitle(rf"Numerical solution of 2D Heat equation", fontsize=30, y=0.98)
fig.tight_layout()
plt.savefig(f"plots/eps/numsolvis_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_Eps_0.0001_N_{2**14}.eps", format="eps")
plt.show()
