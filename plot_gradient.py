import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

# Import your network architecture
from NN_coordination import make_forward_dirichlet_bc

# Force float64 for precision guarantees
jax.config.update("jax_enable_x64", True)

def plot_initial_condition_gradient_norm(domain_type="L"):
    dim = 2
    res_plot = 150  # Dense resolution for a crisp plot
    
    # 1. Initialize the Neural Network Architecture
    nn_forward, param_count = make_forward_dirichlet_bc(dim, 6)
    
    # 2. Load the Saved Parameters
    param_path = f"saved_params/params_heat_initial_dump_{domain_type}"
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Could not find saved parameters at: {param_path}")
        
    with open(param_path, 'rb') as f:
        params = pickle.load(f)
        
    print(f"Successfully loaded parameters from {param_path}")

    # 3. Create a Dense Grid for Plotting
    x_dense = jnp.linspace(-jnp.pi, jnp.pi, res_plot)
    y_dense = jnp.linspace(-jnp.pi, jnp.pi, res_plot)
    X_grid, Y_grid = jnp.meshgrid(x_dense, y_dense)
    
    # Stack into shape (2, N)
    plot_pts = jnp.stack([X_grid.ravel(), Y_grid.ravel()])
    
    # 4. Define the Gradient Function using JAX
    # We define a helper that takes a single point (2, 1) and returns a scalar
    def single_point_forward(x_single):
        return jnp.squeeze(nn_forward(params, x_single))
    
    # Take the gradient of that scalar with respect to the input point
    grad_fn = jax.grad(single_point_forward)
    
    # Vectorize the gradient function across all N points
    # in_axes=1 means we map over the columns of plot_pts
    # out_axes=1 ensures the output gradients are also stacked as columns (2, N)
    batched_grad_fn = jax.vmap(grad_fn, in_axes=1, out_axes=1)
    
    print("Evaluating gradients over the domain...")
    # Compute gradients: shape (2, N)
    grads = batched_grad_fn(plot_pts.reshape(2, -1, 1))
    
    # Compute the L2 norm of the gradient at each point: shape (N,)
    grad_norms = jnp.linalg.norm(grads, axis=0)
    
    # Reshape back to the 2D grid
    Z_grad_norm = grad_norms.reshape(res_plot, res_plot)
    
    # 5. Mask out the missing quadrant for the L-shape
    Z_plot = np.array(Z_grad_norm)
    if domain_type == "L":
        mask_missing_quadrant = (X_grid > 0.0) & (Y_grid > 0.0)
        Z_plot[mask_missing_quadrant] = np.nan

    # 6. Generate the Publication-Ready Plot
    os.makedirs("plot_outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plotting the gradient norm
    mesh = ax.pcolormesh(X_grid, Y_grid, Z_plot, shading='auto', cmap='magma')
    
    # Formatting
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(r'$\|\nabla \text{NN}(x,y)\|_2$ at Initial Condition', fontsize=14)
    
    # Colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Gradient Magnitude', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    file_prefix = f"plot_outputs/initial_gradient_norm_{domain_type}"
    plt.savefig(f"{file_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{file_prefix}.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {file_prefix}.png")

if __name__ == "__main__":
    plot_initial_condition_gradient_norm(domain_type="L")