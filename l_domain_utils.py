import jax
import jax.numpy as jnp
import numpy as np
def l_weights_1d(a, b, n, quad_type):
    """1D composite quadrature weights for n uniform nodes on [a, b].
 
    trapezoid: h * [1/2, 1, ..., 1, 1/2]            (any n >= 2)
    simpson:   h/3 * [1, 4, 2, 4, ..., 2, 4, 1]     (n odd, n >= 3)
    """
    if quad_type == "trapezoid":
        h = (b - a) / (n - 1)
        w = jnp.full(n, h).at[0].set(h / 2).at[-1].set(h / 2)
    elif quad_type == "simpson":
        h = (b - a) / (n - 1)
        w = jnp.zeros(n).at[0::2].set(2 * h / 3).at[1::2].set(4 * h / 3)
        w = w.at[0].set(h / 3).at[-1].set(h / 3)
    else:
        raise ValueError(f"Unknown quad_type '{quad_type}'. Use 'trapezoid' or 'simpson'.")
    return w
 

def square_rule(lower, upper, n, quad_type):
    """Tensor-product nodes (dim, n^dim) and weights (n^dim,) on a box.
 
    lower, upper: array-like of length dim with the box corners.
    Node ordering matches jnp.meshgrid with default indexing, i.e. the
    same ordering as stacking the raveled meshgrid arrays.
    """
    dim = len(lower)
    axes = [jnp.linspace(lower[i], upper[i], n) for i in range(dim)]
    ws_1d = [l_weights_1d(lower[i], upper[i], n, quad_type) for i in range(dim)]
    grid = jnp.meshgrid(*axes)
    wgrid = jnp.meshgrid(*ws_1d)
    nodes = jnp.stack([g.ravel() for g in grid])
    weights = jnp.prod(jnp.stack([w.ravel() for w in wgrid]), axis=0)
    return nodes, weights
 
 
def l_shape_quadrature(n, quad_type, merge_duplicates=True):
    pi = jnp.pi
    vertices = jnp.array(
        [
            [[-pi, -pi], [0.0, 0.0]],  # square 1
            [[-pi, 0.0], [0.0, pi]],   # square 2
            [[0.0, -pi], [pi, 0.0]],   # square 3
        ]
    )
 
    nodes_list, weights_list = [], []
    for v in vertices:
        nodes, weights = square_rule(v[0], v[1], n, quad_type)
        nodes_list.append(nodes)
        weights_list.append(weights)
 
    xx = jnp.concatenate(nodes_list, axis=1)  # (2, 3 n^2)
    w = jnp.concatenate(weights_list)         # (3 n^2,)
 
    if merge_duplicates:
        # The duplicated coordinates are bitwise identical (same linspace
        # calls), so exact comparison in np.unique is safe. Done once at
        # setup with numpy; not intended to run inside jit.
        pts = np.asarray(xx).T
        unique_pts, inverse = np.unique(pts, axis=0, return_inverse=True)
        w = jax.ops.segment_sum(
            w, jnp.asarray(inverse), num_segments=unique_pts.shape[0]
        )
        xx = jnp.asarray(unique_pts.T)
 
    return xx, w