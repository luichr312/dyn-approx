import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt
from NN_coordination import make_forward_dirichlet_bc
from l_domain_utils import l_shape_quadrature
from main import heat_initial_condition_L

def plot_l_shape_solution(resolution_plot=50):
    dim = 2
    
    # 1. Generate the unstructured L-shaped domain points
    xx_plot, _ = l_shape_quadrature(resolution_plot, quad_type="simpson")
    x = xx_plot[0, :]
    y = xx_plot[1, :]
    
    # 2. Recreate the forward network architecture
    # Note: initial_cond_learn uses 8 hidden units for the L-shape
    nn_forward, param_count = make_forward_dirichlet_bc(dim, 8)
    
    # 3. Load the saved parameters from your fitting routine
    try:
        with open("saved_params/params_heat_initial_dump_L", "rb") as f:
            params = pickle.load(f)
    except FileNotFoundError:
        print("Parameter file not found. Ensure initial_cond_learn(domain_type='L') has finished running.")
        return

# 4. Evaluate both the target condition and the learned NN function
    # Use .ravel() to ensure they are strictly 1D arrays of shape (N,)
    target_f = heat_initial_condition_L(xx_plot).ravel()
    learned_f = nn_forward(params, xx_plot).ravel()
    
    # Now this subtraction will happen element-wise correctly
    error = jnp.abs(target_f - learned_f)
    # 5. Plot using tricontourf for unstructured data
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Target Initial Condition
    tc0 = axes[0].tricontourf(x, y, target_f, levels=50, cmap="viridis")
    axes[0].set_title("Target Initial Condition (L-Shape)")
    axes[0].set_aspect('equal')
    fig.colorbar(tc0, ax=axes[0])
    
    # Fitted NN Solution
    tc1 = axes[1].tricontourf(x, y, learned_f, levels=50, cmap="viridis")
    axes[1].set_title("Fitted NN Solution")
    axes[1].set_aspect('equal')
    fig.colorbar(tc1, ax=axes[1])
    
    # Absolute Error
    tc2 = axes[2].tricontourf(x, y, error, levels=50, cmap="magma")
    axes[2].set_title("Absolute Error")
    axes[2].set_aspect('equal')
    fig.colorbar(tc2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_l_shape_solution()