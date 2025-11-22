from NN_coordination import make_forward_dirichlet_bc
from classic_optimizer import train_model_classic
from integrators import IntegratorFittingInitialRK4, ImplicitHeat2D
from jax import random, numpy as jnp
import jax
import matplotlib.pyplot as plt
import pickle
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

def solve_transport_2d_no_bc():
    N = 400
    T = 1
    resolution_plot = 50
    resolution_quad = 50
    d = 2
    vertices = jnp.array([[-jnp.pi,-jnp.pi], [jnp.pi,jnp.pi]])
    eps = 1e-4
    initial_condition = heat_initial_condition
    gauss_steps = 20

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(d)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    f = gaussian(xx_plot).reshape(resolution_plot, resolution_plot)
    plt.pcolormesh(*grid_plot, f, shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()

    nn_forward, param_count = make_forward_dirichlet_bc(d, 5)
    key = random.PRNGKey(0)
    params = jnp.array(random.normal(key, (param_count,1), dtype=jnp.float64))

    params = train_model_classic(nn_forward, params, xx_plot, initial_condition)

    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, eps, initial_condition)
    init_fit.integrate(100,1)
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, init_fit.params, nn_forward, eps, initial_condition)
    init_fit.integrate(100,1)

    with open("params_heat_initial", "wb") as f:
        pickle.dump(init_fit.params, f)

    learned_f = nn_forward(init_fit.params, init_fit.xx_plot)
    print("Error after fitting:", 1 / jnp.sqrt(resolution_plot**2) * jnp.linalg.norm(learned_f - gaussian(xx_plot)))

    plt.pcolormesh(*grid_plot, learned_f.reshape((resolution_plot,resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()

    heat_integrator = ImplicitHeat2D(vertices, xx_plot,resolution_quad, params, nn_forward, eps, gauss_steps)
    heat_integrator.integrate(N,T)

    learned_sol = nn_forward(heat_integrator.params, heat_integrator.xx_plot)
    print((learned_sol-heat_exact_solution(heat_integrator.params, heat_integrator.xx_plot)).shape)
    print("Solution error:", 1 / jnp.sqrt(resolution_plot ** 2) * jnp.linalg.norm(learned_sol - heat_exact_solution(xx_plot,1)))



def test_initial_cond_learn():
    T = 1
    resolution_plot = 50
    resolution_quad = 200
    d = 1
    vertices = [[-jnp.pi], [jnp.pi]]
    eps = 1e-4

    init_condition = gaussian

    xx_plot = jnp.linspace(vertices[0][0], vertices[1][0], resolution_plot).reshape(1, resolution_plot)

    f = init_condition(xx_plot).reshape(1, -1)
    nn_forward, param_count = make_forward_dirichlet_bc(d, 5)


    key = random.PRNGKey(0)
    params = jnp.array(random.normal(key, (param_count, 1), dtype=jnp.float64))

    params = train_model_classic(nn_forward, params, xx_plot, init_condition)

    classic_f = nn_forward(params, xx_plot)
    print("Error after classic opt:",
          1 / jnp.sqrt(resolution_plot) * jnp.linalg.norm(classic_f - f))

    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, eps,
                                           init_condition)
    init_fit.integrate(100, 1)
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, init_fit.params, nn_forward, eps,
                                           init_condition)
    init_fit.integrate(100, 1)

    learned_f = nn_forward(init_fit.params, init_fit.xx_plot)


    print("Error after fitting:",
          1 / jnp.sqrt(resolution_plot) * jnp.linalg.norm(learned_f - f))

    plt.figure()
    plt.plot(xx_plot.reshape(-1), f.reshape(-1), label="exact")
    plt.plot(xx_plot.reshape(-1), classic_f.reshape(-1), label="classic optimiser")
    plt.plot(xx_plot.reshape(-1), learned_f.reshape(-1), label="final")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Comparison")
    plt.legend()
    plt.grid(True)

    plt.show()
if __name__ == "__main__":
    solve_transport_2d_no_bc()
