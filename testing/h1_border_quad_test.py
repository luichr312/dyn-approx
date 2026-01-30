import jax
from jax import jit, random, numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

def make_forward_dirichlet_bc(d, hidden_size):
    # Set up names and shape of parameters
    keys = ["b_input", "W1", "b1", "W_out", "b_out"]
    shapes = [(0, 0), (d, 1), (hidden_size, d), (hidden_size, 1)]
    shapes += [(1, hidden_size), (1, 1)]

    # Index the position of each parameter in a 1D array
    l = np.cumsum([el[0] * el[1] for el in shapes])
    indices = {keys[i]: (jnp.arange(l[i], l[i + 1]), i) for i in range(len(keys))}
    def get_index(k):
        return indices[k][0]
    def get_shape(k):
        return shapes[indices[k][1] + 1]

    # The forwards pass takes parameters as 1D array and uses the indexing above to find retrieve the parameters from
    # in their correct shape
    # All the indexing work will be compiled away on the first run
    # Input: x : (mesh_size, d)
    # Output : (mesh_size, 1)
    def forward(params, x):
        h0 = x + params[get_index("b_input")].reshape(get_shape("b_input"))
        h1 = jnp.tanh(jnp.dot(params[get_index("W1")].reshape(get_shape("W1")), h0)
                      + params[get_index("b1")].reshape(get_shape("b1")))
        out = (jnp.dot(params[get_index("W_out")].reshape(get_shape("W_out")), h1)
                      + params[get_index("b_out")].reshape(get_shape("b_out")))
        return out
    #return forward, l[-1]
    return jit(forward), l[-1]

nn_forward, param_count = make_forward_dirichlet_bc(2,2)
params = jnp.array([1,1,1,1,1,1,1,1,1,2,1], dtype=float).reshape((param_count,1))
print(param_count)

T = 1
tau = 0.1
resolution_quad = 3
dimension = 2
vertices = jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]])
volume = jnp.abs((vertices[1][0]-vertices[0][0]))**dimension

quad_type = "simpson"
num_eval_points_1_axis = {"trapezoid":resolution_quad, "simpson":2*resolution_quad-1}
axes_quad = [jnp.linspace(vertices[0][i], vertices[1][i], num_eval_points_1_axis[quad_type]) for i in range(dimension)]
grid_quad = jnp.meshgrid(*axes_quad)
xx_quad = jnp.stack([g.ravel() for g in grid_quad])
M_quad = resolution_quad**dimension


# Boolean mask which gives all elements corresponding to the border from xx_quad
border_mask = jnp.any(jnp.logical_or(xx_quad == vertices[0].reshape(2, 1),
                                          xx_quad == vertices[1].reshape(2, 1)), axis=0)


if quad_type == "trapezoid":
    weights_1d = jnp.ones(resolution_quad).at[jnp.array([0,-1])].set(0.5).reshape(-1, 1)
elif quad_type == "simpson":
    weights_1d = 1/6*jnp.array([1, 4] + [2 * (i % 2 + 1) for i in range(2*(resolution_quad-2))] + [1]).reshape(1, -1)
else:
    raise ValueError("quad_type must be trapezoid or simpson")

quad_weights_interior = (weights_1d * weights_1d.T).ravel().reshape(-1, 1)

one_border_volume = jnp.abs(vertices[1][0] - vertices[0][0])

border_axes_mask = [jnp.logical_or(xx_quad[i, border_mask] == vertices[0][i],
                                        xx_quad[i, border_mask] == vertices[1][i])
                         for i in range(dimension-1,-1,-1)]


if quad_type == "trapezoid":
    qw_coeff = 2
elif quad_type == "simpson":
    qw_coeff = 6
else:
    raise ValueError("quad_type must be trapezoid or simpson")

quad_weights_border = [qw_coeff * quad_weights_interior[border_mask][border_axes_mask[i]]
                            for i in range(dimension)]

h = one_border_volume / (resolution_quad-1)
h_squared = h * h


def jacobian_x_forward(p, x):
    jac = jax.jacobian(lambda v: nn_forward(p, v.reshape(dimension, 1)))
    jacs = jnp.squeeze(jax.vmap(jac, 1)(x))
    return jacs


jacobians = jacobian_x_forward(params, xx_quad[:, border_mask])
op_mat_jacs = jax.jacobian(jacobian_x_forward)(params, xx_quad[:, border_mask])

#breakpoint()
integ = h*((quad_weights_border[0] * jacobians[border_axes_mask[0], 0].reshape(-1,1)).T
             @ op_mat_jacs[border_axes_mask[0], 0, 6, 0])

print(integ)
'''
h1_semi_contribution = (1.0 / len(border_axes_mask[0]) * one_border_volume*4 / 2.0 *
            ((quad_weights_border[0] * jacobians[border_axes_mask[0], 0].reshape(-1,1)).T
             @ op_mat_jacs[border_axes_mask[0], 0, :, 0]
             + (quad_weights_border[1] * jacobians[border_axes_mask[1], 1].reshape(-1, 1)).T
             @ op_mat_jacs[border_axes_mask[1], 1, :, 0]).T)

print(h1_semi_contribution[8])
'''
h1_semi_contribution = ( h*
            ((quad_weights_border[0] * jacobians[border_axes_mask[0], 0].reshape(-1,1)).T
             @ op_mat_jacs[border_axes_mask[0], 0, :, 0]
             + (quad_weights_border[1] * jacobians[border_axes_mask[1], 1].reshape(-1, 1)).T
             @ op_mat_jacs[border_axes_mask[1], 1, :, 0]).T)

print(h1_semi_contribution[6])
