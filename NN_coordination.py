import jax
from jax import jit, random, numpy as jnp
import numpy as np



def make_forward_dirichlet_bc(d, hidden_size):
    # Set up names and shape of parameters
    keys = ["b_input", "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W_out", "b_out"]
    shapes = [(0, 0), (d, 1), (hidden_size, d), (hidden_size, 1)]
    for _ in range(3):
        shapes += [(hidden_size, hidden_size), (hidden_size, 1)]
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
        h2 = jnp.tanh(jnp.dot(params[get_index("W2")].reshape(get_shape("W2")), h1)
                      + params[get_index("b2")].reshape(get_shape("b2")))
        h3 = jnp.tanh(jnp.dot(params[get_index("W3")].reshape(get_shape("W3")), h2)
                      + params[get_index("b3")].reshape(get_shape("b3")))
        h4 = jnp.tanh(jnp.dot(params[get_index("W4")].reshape(get_shape("W4")), h3)
                      + params[get_index("b4")].reshape(get_shape("b4")))
        out_no_bc = (jnp.dot(params[get_index("W_out")].reshape(get_shape("W_out")), h4)
                      + params[get_index("b_out")].reshape(get_shape("b_out")))
        # breakpoint()
        out = out_no_bc
        # out = out_no_bc*((x[0]-jnp.pi)*(x[0]+jnp.pi)*(x[1]-jnp.pi)*(x[1]+jnp.pi)).reshape(1,-1)
        return out
    # return forward, l[-1]
    return jit(forward), l[-1]


