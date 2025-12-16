import jax
import jax.numpy as jnp
import numpy as np
from fontTools.feaLib.ast import Block
from jax import jit, vmap, lax
from abc import ABC, abstractmethod



class Integrator(ABC):
    # vertices: Vertices of cubic domain that we are working on.
    # resolution_plot: resolution of **ONE axis** of the plot meshgrid
    # M_plot: NUMBER OF POINTS IN THE **WHOLE** MESHGRID
    # same for quad
    # params: parameters of the Neural Network.
    # NN: the architecture of the Neural Network. It provides a .forward method which takes parameters and xx-input.
    # quad_type: "trapezoid" or "simpson"
    def __init__(self, vertices, xx_plot, resolution_quad, params, forward, reg_eps, quad_type="trapezoid"):
        self.params = params
        self.forward = forward

        self.dimension = len(vertices[0])
        self.volume = jnp.abs((vertices[1][0]-vertices[0][0]))**self.dimension

        # Initialise domain grid for plot:
        self.xx_plot = xx_plot
        self.M_plot = xx_plot.shape[1]

        # Initialise domain grid for quadrature:
        # Careful: Resolution of the quadrature ≠ Number of points where f is evaluated
        # resolution_quad gives number of "primary" nodes. For Simpson rule f.ex. num_eval_points_1_axis contains
        # ADDITIONALLY the midpoint for each adjacent "primary" nodes.
        self.resolution_quad = resolution_quad

        self.num_eval_points_1_axis = {"trapezoid":resolution_quad, "simpson":2*resolution_quad-1}
        axes_quad = [jnp.linspace(vertices[0][i], vertices[1][i], self.num_eval_points_1_axis[quad_type]) for i in range(self.dimension)]
        grid_quad = jnp.meshgrid(*axes_quad)
        self.xx_quad = jnp.stack([g.ravel() for g in grid_quad])
        self.M_quad = resolution_quad**self.dimension

        assert(self.dimension == 2)
        # Boolean mask which gives all elements corresponding to the vertices from xx_quad
        self.vertices_mask = jnp.all(jnp.logical_or(self.xx_quad == vertices[0].reshape(2, 1),
                                                    self.xx_quad == vertices[1].reshape(2, 1)), axis=0)

        # Boolean mask which gives all elements corresponding to the border from xx_quad
        self.border_mask = jnp.any(jnp.logical_or(self.xx_quad == vertices[0].reshape(2, 1),
                                                  self.xx_quad == vertices[1].reshape(2, 1)), axis=0)

        self.quad_weights_interior = (jnp.where(self.border_mask, 0.5, 1.0)
                                      * jnp.where(self.vertices_mask, 0.5, 1)).reshape(-1, 1)


        if quad_type == "trapezoid":
            weights_1d = jnp.ones(resolution_quad).at[jnp.array([0,-1])].set(0.5).reshape(-1, 1)
        elif quad_type == "simpson":
            weights_1d = 1/6*jnp.array([1, 4] + [2 * (i % 2 + 1) for i in range(2*(resolution_quad-2))] + [1]).reshape(1, -1)
        else:
            raise ValueError("quad_type must be trapezoid or simpson")

        self.quad_weights_interior = (weights_1d * weights_1d.T).ravel().reshape(-1, 1)

        self.params0 = params
        self.reg_eps = reg_eps

    @abstractmethod
    def make_step(self, time_step):
        pass

    def integrate(self, steps, time_bound, return_values=False):
        err_estimate = 0
        if return_values:
            # Set up array in which the results of the integration will be returned
            vals = np.zeros((self.M_plot, steps + 1))
            vals[:, 0] = self.forward(self.params, self.xx_plot)

        tau = time_bound*1.0/steps
        step = self.make_step(tau)
        for i in range(0, steps):
            #   with jax.profiler.TraceAnnotation("integration_step", step_num=i):
            self.params = step(self.params)# .block_until_ready()
            # if i % 10 == 0:
            #     print(f"integration step {i}")
            if return_values:
                vals[:,i+1]= self.forward(self.params, self.xx_plot)

        if return_values:
            return err_estimate,vals
        return err_estimate

class IntegratorFittingInitialRK4(Integrator):
    def __init__(self, vertices, xx_plot, resolution_quad, params, forward, reg_eps, target_function, quad_type="trapezoid"):
        self.target_function = target_function
        super().__init__(vertices, xx_plot, resolution_quad, params, forward, reg_eps, quad_type)


    def make_step(self, tau):
        def step(params):
            def mass_NGS(p):
                grad = jnp.squeeze(jax.jacobian(self.forward)(p, self.xx_quad))
                mass_mat = self.volume * 1.0 / self.M_quad * grad.T @ (self.quad_weights_interior * grad)
                return mass_mat, grad

            f = self.target_function(self.xx_quad) - self.forward(self.params0, self.xx_quad)

            # Create and solve system for k1
            M_NGS, grad_phi = mass_NGS(params)
            rhs = 1.0/self.M_quad*self.volume* (f @ (self.quad_weights_interior * grad_phi)).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k1 = jnp.linalg.solve(M_reg, rhs)

            # Create and solve system for k2
            M_NGS, grad_phi = mass_NGS(params + 0.5 * tau * k1)
            rhs = 1.0/self.M_quad*self.volume* (f @ (self.quad_weights_interior * grad_phi)).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k2 = jnp.linalg.solve(M_reg, rhs)

            # Create and solve system for k3
            M_NGS, grad_phi = mass_NGS(params + 0.5 * tau * k2)
            rhs = 1.0/self.M_quad*self.volume* (f @ (self.quad_weights_interior * grad_phi)).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k3 = jnp.linalg.solve(M_reg, rhs)

            # Create and solve system for k4
            M_NGS, grad_phi = mass_NGS(params + tau * k3)
            rhs = 1.0/self.M_quad*self.volume* (f @ (self.quad_weights_interior * grad_phi)).T
            M_reg = M_NGS + self.reg_eps ** 2 * jnp.eye(len(rhs))
            k4 = jnp.linalg.solve(M_reg, rhs)

            return params + tau/6*(k1+2*k2+2*k3+k4)

        return jit(step)

class ImplicitHeat2D(Integrator):
    def __init__(self, vertices, xx_plot, resolution_quad, params, forward, reg_eps, gauss_iter_steps, lambda_dampening=0, quad_type="trapezoid"):
        self.gauss_iter_steps = gauss_iter_steps
        self.lambda_dampening = lambda_dampening

        if len(vertices[0]) != 2:
            raise ValueError("dimension must be 2")

        self.border_volume = 4 * jnp.abs(vertices[1][0] - vertices[0][0])
        super().__init__(vertices, xx_plot, resolution_quad, params, forward, reg_eps, quad_type)

        # Boolean mask which gives all elements corresponding to the vertices from xx_quad
        # self.vertices_mask = jnp.all(jnp.logical_or(self.xx_quad == vertices[0].reshape(2, 1),
        #                                            self.xx_quad == vertices[1].reshape(2, 1)), axis=0)

        # The following looks kinda messy, but is useful to implement the H^1 norm on the border

        # Boolean mask which gives all elements corresponding to the border from xx_quad
        # self.border_mask = jnp.any(jnp.logical_or(self.xx_quad == vertices[0].reshape(2, 1),
        #                                          self.xx_quad == vertices[1].reshape(2,1)), axis=0)

        # Boolean mask which gives the position of the points corresponding to given axis **in border_mask**
        # i.e. xx_quad[:,border_mask][:,border_axes_mask[0]] gives all the quad points corresponding to the edges
        # parallel to the x-axis
        self.border_axes_mask = [jnp.logical_or(self.xx_quad[i, self.border_mask] == vertices[0][i],
                                                self.xx_quad[i, self.border_mask] == vertices[1][i])
                                 for i in range(self.dimension-1,-1,-1)]


        # self.quad_weights_interior = (jnp.where(self.border_mask, 0.5, 1.0)
        #                         * jnp.where(self.vertices_mask, 0.5, 1)).reshape(-1, 1)


        if quad_type == "trapezoid":
            self.quad_weights_border = [2*self.quad_weights_interior[self.border_mask][self.border_axes_mask[i]]
                               for i in range(self.dimension)]
        elif quad_type == "simpson":
            self.quad_weights_border = [6*self.quad_weights_interior[self.border_mask][self.border_axes_mask[i]]
                               for i in range(self.dimension)]
        else:
            raise ValueError("quad_type must be trapezoid or simpson")



    def make_step(self, tau):
        def jacobian_x_forward(p, x):
            jac = jax.jacobian(lambda v: self.forward(p, v.reshape(self.dimension, 1)))
            jacs = jnp.squeeze(jax.vmap(jac, 1)(x))
            return jacs

        def laplacian_x_forward(p, x):
            H = jax.hessian(lambda v: self.forward(p, v.reshape(self.dimension, 1)))
            hessians = jnp.squeeze(vmap(H, 1)(x))
            return jnp.trace(hessians, axis1=1, axis2=2)


        def rhs_H1_border_helper(op_mat, op_mat_jacs, params):
            v_border = self.forward(params, self.xx_quad[:, self.border_mask])/tau
            # simple quadrature over all 4 borders at once. No quad_weights needed since periodic
            l2_contribution = 1.0 / len(self.border_mask) * self.border_volume * (v_border @ op_mat).T

            # The following has shape (border quad, 2)
            jacobians =jacobian_x_forward(params, self.xx_quad[:, self.border_mask])/tau

            # The first term does quadrature along the x-direction, the second along y
            h1_semi_contribution = (1.0 / len(self.border_axes_mask[0]) * self.border_volume / 2.0 *
                                    ((self.quad_weights_border[0] * jacobians[self.border_axes_mask[0], 0].reshape(-1,1)).T
                                     @ op_mat_jacs[self.border_axes_mask[0], 0, :, 0]
                                     + (self.quad_weights_border[1] * jacobians[self.border_axes_mask[1], 1].reshape(-1, 1)).T
                                     @ op_mat_jacs[self.border_axes_mask[1], 1, :, 0]).T)
            #h1_semi_contribution = 0*h1_semi_contribution

            return l2_contribution + h1_semi_contribution

        def step(params):

            eval_nodes = 1/tau*jnp.squeeze(jax.jacobian(self.forward)(params, self.xx_quad))
            B = eval_nodes - jnp.squeeze(jax.jacobian(laplacian_x_forward)(params, self.xx_quad))
            B_sys =1.0/self.M_quad*self.volume* B.T @ (self.quad_weights_interior * B)


            eval_nodes_border = eval_nodes[self.border_mask, :]
            D_sys = 1.0/len(self.border_mask) * self.border_volume * eval_nodes_border.T @ eval_nodes_border

            # The following returns (len(border_mask), dimension, len(params), 1) and contains for every point on the
            # border and for every parameter p: [d_p d_1 Phi, d_p d_2 Phi]
            jacobians_at_boundary = 1/tau*jax.jacobian(jacobian_x_forward)(params, self.xx_quad[:, self.border_mask])

            # For every direction we do the quadrature along the corresponding axis of jacobians_at_boundary such that we choose
            # the corresponding partial derivative
            S_sys = 1.0/len(self.border_axes_mask[0]) * self.border_volume/2.0 * (jacobians_at_boundary[self.border_axes_mask[0], 0, :, 0].T @ (
                        self.quad_weights_border[0] * jacobians_at_boundary[self.border_axes_mask[0], 0, :, 0])
                 + jacobians_at_boundary[self.border_axes_mask[1], 1, :, 0].T @ (
                             self.quad_weights_border[1] * jacobians_at_boundary[self.border_axes_mask[1], 1, :, 0]))

            #S_sys = 0*S_sys

            system_matrix = B_sys + D_sys + S_sys + 3/2.0*self.reg_eps**2/tau**2*jnp.eye(S_sys.shape[0])

            c, low = jax.scipy.linalg.cho_factor(system_matrix)
            params0 = params

            Clamdau0 = self.lambda_dampening*rhs_H1_border_helper(eval_nodes_border, jacobians_at_boundary, params0)

            # FIRST ITERATION JUST HAS RHS 0? NO! ALMOST...

            def loop_body(i, carry):
                p, dsq = carry
                diff_quot = 1/tau*(self.forward(p, self.xx_quad) - self.forward(params0, self.xx_quad))
                # THIS DOESN'T NEED TO BE COMPUTED ON FIRST PASS
                fu1k = laplacian_x_forward(p, self.xx_quad).reshape(1,-1)
                r1k = diff_quot - fu1k
                Br1k = 1.0/self.M_quad*self.volume * ((self.quad_weights_interior.T * r1k) @ B).T

                Cuk = rhs_H1_border_helper(eval_nodes_border, jacobians_at_boundary, p)
                Cr2k = Cuk - Clamdau0

                rhs_reg =  + 1/2.0*self.reg_eps**2/tau**2*(p-params0)
                rhs = -Br1k - Cr2k - rhs_reg
                p_update = jax.scipy.linalg.cho_solve((c,low), rhs)

                # def compute_dsq():
                #     d = 1.0 / self.M_quad* self.volume * jnp.sum(((B @ p_update).T + r1k) ** 2)
                #     d += jnp.sum(p_update** 2)  * self.reg_eps ** 2 / tau ** 2
                #     d += jnp.sum((p - params0 + p_update)**2) * 1 / 2 * self.reg_eps ** 2 / tau ** 2
                #     '''
                #     missing H1 stuff....
                #     '''
                #     return 0
                # dsq = lax.cond(i == self.gauss_iter_steps-1, lambda _: compute_dsq(), lambda _: dsq, None)
                return p + p_update, dsq

            delta_sq = 0
            params, delta_sq = lax.fori_loop(0, self.gauss_iter_steps, loop_body, (params, delta_sq))

            return params
        return jit(step)

