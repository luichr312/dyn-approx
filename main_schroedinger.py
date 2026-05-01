import os.path

from NN_coordination import make_forward, make_forward_schroedinger_complex
from classic_optimizer import train_model_classic
from integrators import IntegratorFittingInitialRK4, ImplicitHeat2D, ImplicitSchroedinger2D
from jax import random, numpy as jnp
import jax
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

jax.config.update("jax_enable_x64", True)





def V_0(x):
    return jnp.zeros_like(x)

def schroedinger_sine_initial(x):
    return (jnp.sin(x[0] / 2 + jnp.pi / 2) * jnp.sin(x[1] / 2 + jnp.pi / 2)).astype(jnp.complex128)

def schroedinger_sine_exact_sol(x, t):
    return jnp.exp(-1j* t / 2) * jnp.sin(x[0] / 2 + jnp.pi / 2) * jnp.sin(x[1] / 2 + jnp.pi / 2)

omega = 2
def V_harmonic(x):
    return omega*omega*(x[0]*x[0]+x[1]*x[1])
def schroedinger_stationary_initial(x):
   return jnp.exp(-omega*(x[0]**2+x[1]**2)/2)
def schroedinger_stationary_exact_sol(x, t):
    return jnp.exp(-2*1j*omega*t)*schroedinger_stationary_initial(x)

def initial_cond_learn(resolution_quad=20):
    resolution_plot = 50
    dim = 2
    vertices = jnp.array([[-jnp.pi,-jnp.pi], [jnp.pi,jnp.pi]])

    initial_condition = schroedinger_stationary_initial

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(dim)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    nn_forward, param_count = make_forward_schroedinger_complex(dim, 6)
    key = random.PRNGKey(0)
    print(param_count)
    params = jnp.array(random.normal(key, (param_count, 1), dtype=jnp.float64))
    #xx_plot_np = np.array(xx_plot)
    # _ = nn_forward(params, xx_plot)
    imaginary_noise = 1e-6j
    params = train_model_classic(nn_forward, params, xx_plot, lambda x: initial_condition(x)+imaginary_noise,
                                 epochs=10000, complex=True)
    N = 800

    eps = 0.000005
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, eps,
                                           initial_condition, complex=True, quad_type='simpson')
    init_fit.integrate(N, 1)

    params = init_fit.params
    learned_f = (nn_forward(params, xx_plot)[0] + 1j*nn_forward(params,xx_plot)[1]).block_until_ready()

    with open("saved_params/schroedinger_stationary/params_initial_dump_more_depth_hs6", "wb") as f:
        pickle.dump(params, f)

    print("Error after fitting:",
          2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_f - initial_condition(xx_plot)))

    plt.pcolormesh(*grid_plot, jnp.abs(jnp.real(initial_condition(xx_plot) - learned_f)).reshape((resolution_plot, resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()

def solve_schroedinger_2d_dirichlet_bc():
    N = 1
    T = 0.0001
    resolution_plot = 100
    resolution_quad = 20
    d = 2
    vertices = jnp.array([[-jnp.pi,-jnp.pi], [jnp.pi,jnp.pi]])
    eps = 0.01
    initial_condition = schroedinger_sine_initial
    exact_sol = schroedinger_sine_exact_sol

    gauss_steps = 20

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])


    nn_forward, param_count = make_forward_schroedinger_complex(d, 6)
    print(param_count)

    path = 'saved_params/schroedinger_sine/params_schroedinger_hs6_imnoise_1e-5_N200_ee-5_N400_ee-6_simpson25_acc_0.003.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            print(params.shape)
    else:
        raise ValueError("No saved params found")

    learned_ini_cond_split = nn_forward(params, xx_plot).block_until_ready()
    learned_ini_cond = learned_ini_cond_split[0] + 1j*learned_ini_cond_split[1]
    print("Error of loaded initial condition:", 2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_ini_cond - initial_condition(xx_plot)))


    integrator = ImplicitSchroedinger2D(vertices, xx_plot,resolution_quad, params, nn_forward, eps, gauss_steps,
                                        potential=V_0, lambda_dampening=0, quad_type='simpson', alpha=0.2)

    integrator.integrate(N,T)

    #with open("saved_params/schroedinger_stat_sol/tmp.pickle", "wb") as f:
    #    pickle.dump(integrator.params, f)

    learned_sol_split = nn_forward(integrator.params, integrator.xx_plot)
    learned_sol = learned_sol_split[0] + 1j*learned_sol_split[1]
    print("Solution error:", 2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_sol - exact_sol(xx_plot,T)))

    abs_error = jnp.abs(learned_sol - exact_sol(xx_plot,1))
    plt.pcolormesh(*grid_plot, abs_error.reshape((resolution_plot, resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()



def test():
    resolution_plot = 50
    d = 2
    vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])
    initial_condition = schroedinger_stationary_initial
    exact_sol = schroedinger_stationary_exact_sol


    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    nn_forward, param_count = make_forward_schroedinger_complex(d, 6)

    path = 'saved_params/schroedinger_stationary/params_schroedinger_stat_omega2_hs6_depth+1_imnoise_1e-5_N800_e5e-6_N1000_e5e-6_N2000_e1e-6_simpson25_acc_0.00058.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            print(params.shape)
    else:
        raise ValueError("No saved params found")

    learned_f_split = nn_forward(params, xx_plot)
    learned_f = learned_f_split[0] + 1j * learned_f_split[1]
    print(jnp.average(learned_f_split[0]), jnp.average(learned_f_split[1]))
    plt.pcolormesh(*grid_plot, learned_f_split[0].reshape(
        (resolution_plot, resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()

    plt.pcolormesh(*grid_plot, learned_f_split[1].reshape(
        (resolution_plot, resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()

def initial_cond_improve():
    path = 'saved_params/schroedinger_stationary/params_schroedinger_stat_omega2_hs6_depth+1_imnoise_1e-5_N800_e5e-6_N1000_e5e-6_simpson25_acc_0.0046.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            print(params.shape)
    else:
        raise ValueError("No saved params found")

    resolution_plot = 50
    resolution_quad = 25
    dim = 2
    vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])

    initial_condition = schroedinger_stationary_initial

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(dim)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    nn_forward, param_count = make_forward_schroedinger_complex(dim, 6)
    print(param_count)
    # xx_plot_np = np.array(xx_plot)
    # _ = nn_forward(params, xx_plot)

    learned_ini_cond_split = nn_forward(params, xx_plot)
    learned_ini_cond = learned_ini_cond_split[0] + 1j * learned_ini_cond_split[1]
    print("Error after fitting:",
          2 * jnp.pi / resolution_plot * jnp.linalg.norm(learned_ini_cond - initial_condition(xx_plot)))

    N = 2000
    eps = 0.000001
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, eps,
                                           initial_condition, complex=True, quad_type='simpson')
    init_fit.integrate(N, 1)

    params = init_fit.params
    learned_f = (nn_forward(params, xx_plot)[0] + 1j * nn_forward(params, xx_plot)[1]).block_until_ready()

    with open("saved_params/schroedinger_stationary/params_initial_dump_more_depth_hs6", "wb") as f:
        pickle.dump(params, f)

    print("Error after fitting:",
          2 * jnp.pi / resolution_plot * jnp.linalg.norm(learned_f - initial_condition(xx_plot)))




if __name__ == "__main__":
    solve_schroedinger_2d_dirichlet_bc()