from jax import numpy as jnp, jit, jacobian


def mass_NGS(forward, params, xx, domain_volume):
    grad_phi = jnp.squeeze(jacobian(forward)(params, xx))
    M_quad = xx.shape[1]
    mass_mat = domain_volume * 1.0 / M_quad * grad_phi.T @ grad_phi
    return mass_mat, grad_phi