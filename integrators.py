import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from NGS_helpers import mass_NGS
from abc import ABC, abstractmethod

class Integrator(ABC):
    # vertices: Vertices of cubic domain that we are working on.
    # resolution_plot: resolution of **ONE axis** of the plot meshgrid
    # M_plot: NUMBER OF POINTS IN THE **WHOLE** MESHGRID
    # same for quad
    # params: parameters of the Neural Network.
    # NN: the architecture of the Neural Network. It provides a .forward method which takes parameters and xx-input.

    def __init__(self, vertices, xx_plot, resolution_quad, params, forward, reg_eps):
        self.params = params
        self.forward = forward

        self.dimension = len(vertices[0])
        self.volume = jnp.abs((vertices[0][0]-vertices[1][0]))**self.dimension

        # Initialise domain grid for plot:
        self.xx_plot = xx_plot
        self.M_plot = xx_plot.shape[0]

        # Initialise domain grid for quadrature:
        self.resolution_quad = resolution_quad
        axes_quad = [jnp.linspace(vertices[0][i], vertices[1][i], resolution_quad) for i in range(self.dimension)]
        grid_quad = jnp.meshgrid(*axes_quad)
        self.xx_quad = jnp.stack([g.ravel() for g in grid_quad], axis=-1).T
        self.M_quad = resolution_quad**self.dimension

        self.reg_eps = reg_eps

    @abstractmethod
    def make_step(self, time_step):
        pass

    def integrate(self, steps, time_bound, return_values=False):
        print(self.volume, self.xx_quad.shape)
        err_estimate = 0
        if return_values:
            # Set up array in which the results of the integration will be returned
            vals = np.zeros((self.M_plot, steps + 1))
            vals[:, 0] = self.forward(self.params, self.xx_plot)

        tau = time_bound*1.0/steps
        step = self.make_step(tau)
        for i in range(0, steps):
            self.params = step(self.params)
            if return_values:
                vals[:,i+1]= self.forward(self.params, self.xx_plot)
            if i % 10== 0:
                print(f"Integration step {i} done")

        if return_values:
            return err_estimate,vals
        return err_estimate

class IntegratorFittingInitialRK4(Integrator):
    def __init__(self, vertices, resolution_plot, resolution_quad, params, forward, reg_eps, target_function):
        self.target_function = target_function
        super().__init__(vertices, resolution_plot, resolution_quad, params, forward, reg_eps)


    def make_step(self, tau):
        def step(params):
            f = self.target_function(self.xx_quad) - self.forward(params, self.xx_quad)

            # Create and solve system for k1
            M_NGS, grad_phi = mass_NGS(self.forward, params, self.xx_quad, self.volume)
            print(M_NGS.shape)
            print(grad_phi.shape)
            rhs = 1.0/self.M_quad*self.volume* (f @ grad_phi).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k1 = jnp.linalg.solve(M_reg, rhs)

            # Create and solve system for k2
            M_NGS, grad_phi = mass_NGS(self.forward, params + 0.5 * tau * k1, self.xx_quad, self.volume)
            rhs = 1.0 / self.M_quad * self.volume * (f @ grad_phi).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k2 = jnp.linalg.solve(M_reg, rhs)

            # Create and solve system for k3
            M_NGS, grad_phi = mass_NGS(self.forward, params + 0.5 * tau * k2, self.xx_quad, self.volume)
            rhs = 1.0 / self.M_quad * self.volume * (f @ grad_phi).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k3 = jnp.linalg.solve(M_reg, rhs)

            # Create and solve system for k4
            M_NGS, grad_phi = mass_NGS(self.forward, params + tau * k3, self.xx_quad, self.volume)
            rhs = 1.0 / self.M_quad * self.volume * (f @ grad_phi).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k4 = jnp.linalg.solve(M_reg, rhs)

            return params + tau/6*(k1+2*k2+2*k3+k4)

        return jit(step)



