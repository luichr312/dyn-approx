import os.path

from NN_coordination import make_forward_dirichlet_bc
from classic_optimizer import train_model_classic
from integrators import IntegratorFittingInitialRK4, ImplicitHeat2D
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

def gaussian(x):
    return jnp.exp(-jnp.linalg.norm(x, axis=0)**2)

def triangle_1d(x):
    return jnp.maximum(0, -2*jnp.abs(x)+1)


def heat_initial_condition(x):
    return jnp.sin(x[0] / 2 - jnp.pi / 2) * jnp.sin(x[1] / 2 - jnp.pi / 2)

def heat_exact_solution(x, t):
    return jnp.exp(-t / 2) * jnp.sin(x[0] / 2 - jnp.pi / 2) * jnp.sin(x[1] / 2 - jnp.pi / 2)

def solve_heat_2d_dirichlet_bc():
    N = 256
    T = 1
    resolution_plot = 50
    resolution_quad = 10
    d = 2
    vertices = jnp.array([[-jnp.pi,-jnp.pi], [jnp.pi,jnp.pi]])
    eps = 0.0001
    initial_condition = heat_initial_condition
    gauss_steps = 20

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])


    nn_forward, param_count = make_forward_dirichlet_bc(d, 6)
    key = random.PRNGKey(0)
    params = jnp.array(random.normal(key, (param_count,1), dtype=jnp.float64))

    if os.path.exists('saved_params/params_heat_initial_hs6_3e-5.pickle'):
        with open('saved_params/params_heat_initial_hs6_3e-5.pickle', 'rb') as f:
            params = pickle.load(f)
    else:
        params = train_model_classic(nn_forward, params, xx_plot, initial_condition)
        init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, eps, initial_condition)
        init_fit.integrate(100,1)
        init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, init_fit.params, nn_forward, eps, initial_condition)
        init_fit.integrate(100,1)

        params = init_fit.params
        with open("saved_params/params_heat_useless.pickle", "wb") as f:
            pickle.dump(params, f)

    learned_f = nn_forward(params, xx_plot)
    print("Error after fitting:", 2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_f - initial_condition(xx_plot)))

    plt.pcolormesh(*grid_plot, learned_f.reshape((resolution_plot,resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()

    heat_integrator = ImplicitHeat2D(vertices, xx_plot,resolution_quad, params, nn_forward, eps, gauss_steps,
                                     lambda_dampening=1, quad_type='simpson')

    heat_integrator.integrate(N,T)
    learned_sol = nn_forward(heat_integrator.params, heat_integrator.xx_plot)
    print("Solution error:", 2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_sol - heat_exact_solution(xx_plot,1)))

    plt.pcolormesh(*grid_plot, learned_sol.reshape((resolution_plot, resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()


def initial_cond_learn(resolution_quad=20, eps=1e-2, N=200):
    resolution_plot = 50
    dim = 2
    vertices = jnp.array([[-jnp.pi,-jnp.pi], [jnp.pi,jnp.pi]])

    initial_condition = heat_initial_condition

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(dim)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    nn_forward, param_count = make_forward_dirichlet_bc(dim, 6)
    key = random.PRNGKey(0)

    # DATA TYPE!!!!
    params = jnp.array(random.normal(key, (param_count, 1), dtype=jnp.float32))
    #xx_plot_np = np.array(xx_plot)
    # _ = nn_forward(params, xx_plot)

    params = train_model_classic(nn_forward, params, xx_plot, initial_condition, epochs=13000)
    eps = 0.00001
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, eps,
                                           initial_condition, quad_type='simpson')
    init_fit.integrate(N, 1)
    N=400
    eps = 0.000001
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, init_fit.params, nn_forward, eps,
                                           initial_condition, quad_type='simpson')
    init_fit.integrate(N, 1)

    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, init_fit.params, nn_forward, eps,
                                           initial_condition, quad_type='simpson')
    init_fit.integrate(N, 1)

    params = init_fit.params
    learned_f = nn_forward(params, xx_plot).block_until_ready()

    with open("saved_params/params_heat_initial_better_hs_6", "wb") as f:
        pickle.dump(params, f)

    print("Error after fitting:",
          2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_f - initial_condition(xx_plot)))

    plt.pcolormesh(*grid_plot, (initial_condition(xx_plot) - learned_f).reshape((resolution_plot, resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()


def convergence_analysis(alpha):
    N = 2 ** np.arange(4, 15)
    EPS = [0.1, 0.01, 0.001, 0.0001]
    print("N: ", N, "EPS: ", EPS)
    T = 1
    resolution_plot = 50
    resolution_quad = 10
    d = 2
    vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])
    initial_condition = heat_initial_condition
    gauss_steps = 20
    lambda_damp = 0.5

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    nn_forward, param_count = make_forward_dirichlet_bc(d, 6)
    key = random.PRNGKey(0)
    params = jnp.array(random.normal(key, (param_count, 1), dtype=jnp.float64))

    # init_fit_eps = 1e-4
    # params = train_model_classic(nn_forward, params, xx_plot, initial_condition)
    # init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, init_fit_eps,
    #                                        initial_condition)
    # init_fit.integrate(100, 1)
    # init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, init_fit.params, nn_forward, init_fit_eps,
    #                                        initial_condition)
    # init_fit.integrate(100, 1)

    #params = init_fit.params
    if os.path.exists('saved_params/params_heat_initial_hs6_3e-5.pickle'):
        with open('saved_params/params_heat_initial_hs6_3e-5.pickle', 'rb') as f:
            params = pickle.load(f)
    else:
        raise ValueError('params_heat_initial_now.pickle does not exist')

    learned_f = nn_forward(params, xx_plot)
    print("Error after fitting:",
          (2*jnp.pi) / resolution_plot * jnp.linalg.norm(learned_f - initial_condition(xx_plot)))

    err = np.zeros((len(EPS), len(N)))
    err_estimate = np.zeros((len(EPS), len(N)))
    for e in range(len(EPS)):
        for n in range(len(N)):
            print(f"--- N={N[n]}, eps={EPS[e]}, Mquad={resolution_quad} ---")
            heat_integrator = ImplicitHeat2D(vertices, xx_plot, resolution_quad, params, nn_forward, EPS[e], gauss_steps,
                                             lambda_dampening=lambda_damp, quad_type='simpson',alpha=alpha)
            heat_integrator.integrate(N[n], T)

            learned_sol = nn_forward(heat_integrator.params, heat_integrator.xx_plot)
            err[e,n] = 2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_sol - heat_exact_solution(xx_plot, T))

        scipy.io.savemat(f'error_mats/errors_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_simpson_10_N_4-15_Eps_0.1-0.01-0.001-0.0001.mat', {
            'errors': err[:e + 1, :],
            'errors_est': err_estimate[:e + 1, :],
            'N_s': N
        })
        print(f"Error: {err}")

def save_sol(alpha, times=None):
    N = 2 ** np.arange(4, 15)
    EPS = [0.1, 0.01, 0.001, 0.0001]
    print("N: ", N, "EPS: ", EPS)
    T = 1
    resolution_quad = 10
    d = 2
    vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])
    gauss_steps = 20
    lambda_damp = 0

    resolution_plot = 50
    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    nn_forward, param_count = make_forward_dirichlet_bc(d, 6)
    if os.path.exists('saved_params/params_heat_initial_hs6_3e-5.pickle'):
        with open('saved_params/params_heat_initial_hs6_3e-5.pickle', 'rb') as f:
            params = pickle.load(f)
    else:
        raise ValueError('params_heat_initial_now.pickle does not exist')
    save_frames = 4
    saved_params = np.zeros((len(EPS), len(N), param_count, save_frames))
    for e in range(len(EPS)):
        for n in range(len(N)):
            print(f"--- N={N[n]}, eps={EPS[e]}, Mquad={resolution_quad} ---")
            heat_integrator = ImplicitHeat2D(vertices, xx_plot, resolution_quad, params, nn_forward, EPS[e], gauss_steps,
                                             lambda_dampening=lambda_damp, quad_type='simpson',alpha=alpha)

            _, saved_params[e, n] = heat_integrator.integrate(N[n], True, save_frames)

        scipy.io.savemat(
            f'saved_sols/saved_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_simpson_10_N_4-15_Eps_0.1-0.01-0.001-0.0001_T_0.25-0.5-0.75-1.mat',
            {
                'params': saved_params[:e + 1],
                'N_s': N
            })

if __name__ == "__main__":
    save_sol(0)
    save_sol(0.2)
    #convergence_analysis(1)

