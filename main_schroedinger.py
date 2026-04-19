import os.path

from NN_coordination import make_forward, make_forward_schroedinger_complex
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



def V_harmonic(x):
    k = 1
    return k*k/2*(x[0]*x[0]+x[1]*x[1])

def V_0(x):
    return jnp.zeros_like(x)

def schroedinger_sine_initial(x):
    return (jnp.sin(x[0] / 2 + jnp.pi / 2) * jnp.sin(x[1] / 2 + jnp.pi / 2))+1e-5j
    return (jnp.sin(x[0] / 2 + jnp.pi / 2) * jnp.sin(x[1] / 2 + jnp.pi / 2)).astype(jnp.complex128)

def schroedinger_sine_exact_sol(x, t):
    return jnp.exp(-1j* t / 2) * jnp.sin(x[0] / 2 + jnp.pi / 2) * jnp.sin(x[1] / 2 + jnp.pi / 2)

def initial_cond_learn(resolution_quad=25):
    resolution_plot = 50
    dim = 2
    vertices = jnp.array([[-jnp.pi,-jnp.pi], [jnp.pi,jnp.pi]])

    initial_condition = schroedinger_sine_initial

    axes_plot = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_plot) for i in range(dim)]
    grid_plot = jnp.meshgrid(*axes_plot)
    xx_plot = jnp.stack([g.ravel() for g in grid_plot])

    nn_forward, param_count = make_forward_schroedinger_complex(dim, 7)
    key = random.PRNGKey(0)
    print(param_count)
    params = jnp.array(random.normal(key, (param_count, 1), dtype=jnp.float64))
    #xx_plot_np = np.array(xx_plot)
    # _ = nn_forward(params, xx_plot)

    params = train_model_classic(nn_forward, params, xx_plot, initial_condition, epochs=10000, complex=True)
    N = 200
    eps = 0.00001
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, params, nn_forward, eps,
                                           initial_condition, complex=True, quad_type='simpson')
    init_fit.integrate(N, 1)
    print("First done")
    N = 400
    eps = 0.000001
    init_fit = IntegratorFittingInitialRK4(vertices, xx_plot, resolution_quad, init_fit.params, nn_forward, eps,
                                           initial_condition, complex=True, quad_type='simpson')
    init_fit.integrate(N, 1)
    print("Second done")


    params = init_fit.params
    learned_f = (nn_forward(params, xx_plot)[0] + 1j*nn_forward(params,xx_plot)[1]).block_until_ready()

    with open("saved_params/params_heat_initial_dump", "wb") as f:
        pickle.dump(params, f)

    print("Error after fitting:",
          2*jnp.pi / resolution_plot * jnp.linalg.norm(learned_f - initial_condition(xx_plot)))

    plt.pcolormesh(*grid_plot, jnp.abs(jnp.real(initial_condition(xx_plot) - learned_f)).reshape((resolution_plot, resolution_plot)), shading='auto', cmap="viridis")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    initial_cond_learn()