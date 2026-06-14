import jax
import jax.numpy as jnp
import numpy as np

# For the l-shaped domain the quad points are ordered x-wise
def l_weights_1d(n, quad_type):
    if quad_type == "trapezoid":
        w = jnp.full(n, 1.0).at[0].set(0.5).at[-1].set(0.5)
    elif quad_type == "simpson":
        w = jnp.zeros(n).at[0::2].set(2.0 / 6.0).at[1::2].set(4.0 / 6.0)
        w = w.at[0].set(1.0 / 6.0).at[-1].set(1.0 / 6.0)
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
    ws_1d = [l_weights_1d(n, quad_type) for i in range(dim)]

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

def l_border_weights(n, quad_type):
    """Closed-boundary nodes (2, B) and 1D weights (B,) for the L-shape."""
    pi = jnp.pi
    # (fixed_axis, fixed_value, a, b): 8 segments of length pi
    segments = [
        (1, -pi, -pi, 0.0), (1, -pi, 0.0,  pi),   # bottom  y = -pi
        (0,  pi, -pi, 0.0),                        # right   x = +pi
        (1, 0.0,  0.0,  pi),                       # reentrant horizontal y = 0
        (0, 0.0,  0.0,  pi),                       # reentrant vertical   x = 0
        (1,  pi, -pi, 0.0),                        # top     y = +pi
        (0, -pi, -pi, 0.0), (0, -pi, 0.0,  pi),   # left    x = -pi
    ]
    pts, wts = [], []
    for axis, val, a, b in segments:
        t = jnp.linspace(a, b, n)
        p = jnp.zeros((2, n)).at[axis].set(val).at[1 - axis].set(t)
        pts.append(p)
        wts.append(l_weights_1d(a, b, n, quad_type))
    xx_b = jnp.concatenate(pts, axis=1)
    w_b = jnp.concatenate(wts)
    # merge shared segment endpoints, summing their weights
    uniq, inv = np.unique(np.asarray(xx_b).T, axis=0, return_inverse=True)
    w_b = jax.ops.segment_sum(w_b, jnp.asarray(inv), num_segments=len(uniq))
    return jnp.asarray(uniq.T), w_b
"""
def l_border_weights(n, quad_type):
    pi = jnp.pi
    vertices = [
        [-pi,-pi],
        [-pi,pi],
        [0,pi],
        [0,0],
        [pi,0],
        [pi,-pi]
    ]
"""