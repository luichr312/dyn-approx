import jax
import jax.numpy as jnp
import numpy as np
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

        # self.quad_weights_interior = (jnp.where(self.border_mask, 0.5, 1.0)
        #                             * jnp.where(self.vertices_mask, 0.5, 1)).reshape(-1, 1)


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

    def integrate(self, steps, time_bound, return_values=False, save_frames=4):
        err_estimate = 0
        if return_values:
            # Set up array in which the results of the integration will be returned
            saved_p = np.zeros((self.params.shape[0], save_frames))

        tau = time_bound*1.0/steps
        step = self.make_step(tau)
        save_interval = steps//save_frames
        if not steps%save_interval == 0:
            print(steps, save_frames,save_interval)
            raise ValueError("save_frames must be compatible with number of steps")
        for i in range(0, steps):
            #   with jax.profiler.TraceAnnotation("integration_step", step_num=i):
            self.params = step(self.params)# .block_until_ready()
            #if i % 1000 == 0:
            #     print(f"integration step {i}")
            if return_values and (i+1)%save_interval == 0:
                saved_p[:,(i+1)//save_interval-1]= self.params.ravel()

        if return_values:
            return err_estimate, saved_p
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
    def __init__(self, vertices, xx_plot, resolution_quad, params, forward, reg_eps, gauss_iter_steps, lambda_dampening=0, quad_type="trapezoid", alpha=1):
        self.gauss_iter_steps = gauss_iter_steps
        self.lambda_dampening = lambda_dampening

        if len(vertices[0]) != 2:
            raise ValueError("dimension must be 2")

        self.one_border_volume = jnp.abs(vertices[1][0] - vertices[0][0])
        super().__init__(vertices, xx_plot, resolution_quad, params, forward, reg_eps, quad_type)

        # Boolean mask which gives the position of the points corresponding to given axis **in border_mask**
        # i.e. xx_quad[:,border_mask][:,border_axes_mask[0]] gives all the quad points corresponding to the edges
        # parallel to the x-axis
        self.border_axes_mask = [jnp.logical_or(self.xx_quad[i, self.border_mask] == vertices[0][i],
                                                self.xx_quad[i, self.border_mask] == vertices[1][i])
                                 for i in range(self.dimension-1,-1,-1)]

        if quad_type == "trapezoid":
            self.qw_coeff = 2
        elif quad_type == "simpson":
            self.qw_coeff = 6
        else:
            raise ValueError("quad_type must be trapezoid or simpson")

        self.quad_weights_border = [self.qw_coeff * self.quad_weights_interior[self.border_mask][self.border_axes_mask[i]]
                                    for i in range(self.dimension)]

        self.quad_weights_border_periodic = (jnp.where(self.vertices_mask[self.border_mask], 2, 1).reshape(-1, 1)
                                        *self.qw_coeff * self.quad_weights_interior[self.border_mask])
        self.h = self.one_border_volume / (resolution_quad-1)
        self.h_squared = self.h * self.h
        self.alpha = alpha

    def make_step(self, tau):
        def jacobian_x_forward(p, x):
            jac = jax.jacobian(lambda v: self.forward(p, v.reshape(self.dimension, 1)))
            jacs = jnp.squeeze(jax.vmap(jac, 1)(x))
            return jacs

        def laplacian_x_forward(p, x):
            H = jax.hessian(lambda v: self.forward(p, v.reshape(self.dimension, 1)))
            hessians = jnp.squeeze(vmap(H, 1)(x))
            return jnp.trace(hessians, axis1=1, axis2=2)


        def rhs_H1_border_helper(eval_dparam_border, eval_dparam_dx_border, params_stage):
            eval_border_stage = self.forward(params_stage, self.xx_quad[:, self.border_mask]) / tau
            # quadrature over all 4 borders at once
            l2_contribution = self.h * (self.quad_weights_border_periodic.T * eval_border_stage @ eval_dparam_border).T

            # The following has shape (border quad, dim)
            eval_dx_border_stage = jacobian_x_forward(params_stage, self.xx_quad[:, self.border_mask]) / tau

            # The first term does quadrature along the x-direction, the second along y
            #h1_semi_contribution = (1.0 / len(self.border_axes_mask[0]) * self.one_border_volume*4*

            h1_semi_contribution = self.alpha*(self.h *
                    ((self.quad_weights_border[0].T * eval_dx_border_stage[self.border_axes_mask[0], 0])
                     @ eval_dparam_dx_border[self.border_axes_mask[0], 0, :, 0]
                     + (self.quad_weights_border[1].T * eval_dx_border_stage[self.border_axes_mask[1], 1])
                     @ eval_dparam_dx_border[self.border_axes_mask[1], 1, :, 0]).T)
            #h1_semi_contribution = 0*h1_semi_contribution

            return l2_contribution + h1_semi_contribution

        def step(params):

            eval_dparam = 1/tau*jnp.squeeze(jax.jacobian(self.forward)(params, self.xx_quad))
            eval_impl_euler = eval_dparam - jnp.squeeze(jax.jacobian(laplacian_x_forward)(params, self.xx_quad))
            sys_l2_interior = self.h_squared * eval_impl_euler.T @ (self.quad_weights_interior * eval_impl_euler)

            eval_dparam_border = eval_dparam[self.border_mask, :]
            sys_l2_border = (self.h * eval_dparam_border.T @(self.quad_weights_border_periodic * eval_dparam_border))

            # The following returns (num of border points, dimension, len(params), 1) and contains for every point on the
            # border and for every parameter params: [d_p d_1 Phi, d_p d_2 Phi]
            eval_dparam_dx_border = 1/tau*jax.jacobian(jacobian_x_forward)(params, self.xx_quad[:, self.border_mask])

            # For every direction we do the quadrature along the corresponding axis of eval_dparam_dx_border such that we choose
            # the corresponding partial derivative
            # sys_h1_border = 1.0 / len(self.border_axes_mask[0]) * self.one_border_volume *4 / 2.0 * (
            #                              eval_dparam_dx_border[self.border_axes_mask[0], 0, :, 0].T @ (
            sys_h1_border = self.alpha * self.h * (eval_dparam_dx_border[self.border_axes_mask[0], 0, :, 0].T @ (
                      self.quad_weights_border[0] * eval_dparam_dx_border[self.border_axes_mask[0], 0, :, 0])
                    + eval_dparam_dx_border[self.border_axes_mask[1], 1, :, 0].T @ (
                       self.quad_weights_border[1] * eval_dparam_dx_border[self.border_axes_mask[1], 1, :, 0]))

            #sys_h1_border = 0*sys_h1_border

            system_matrix = (sys_l2_interior + sys_l2_border + sys_h1_border
                             + 3/2.0*self.reg_eps**2/tau**2*jnp.eye(sys_h1_border.shape[0]))

            c, low = jax.scipy.linalg.cho_factor(system_matrix)
            params0 = params

            rhs_border_0_lambda = self.lambda_dampening*rhs_H1_border_helper(eval_dparam_border, eval_dparam_dx_border, params0)

            # FIRST ITERATION JUST HAS RHS 0? NO! ALMOST...
            # Consider removing lax.fori_loop and just use regular python loop. XLA optimising not as aggressive inside
            # lax stuff.
            '''
            def loop_body(i, carry):
                params, dsq = carry
                diff_quot = 1/tau*(self.forward(params, self.xx_quad) - self.forward(params0, self.xx_quad))
                # THIS DOESN'T NEED TO BE COMPUTED ON FIRST PASS
                eval_laplacian_stage = laplacian_x_forward(params, self.xx_quad).reshape(1,-1)
                rhs_eval_l2_interior = diff_quot - eval_laplacian_stage
                rhs_l2_interior = self.h_squared * ((self.quad_weights_interior.T * rhs_eval_l2_interior) @ eval_impl_euler).T

                rhs_border_stage = rhs_H1_border_helper(eval_dparam_border, eval_dparam_dx_border, params)
                rhs_border = rhs_border_stage - rhs_border_0_lambda

                rhs_reg = 1/2.0*self.reg_eps**2/tau**2*(params-params0)
                rhs = -rhs_l2_interior - rhs_border - rhs_reg
                p_update = jax.scipy.linalg.cho_solve((c,low), rhs)

                # def compute_dsq():
                #     d = 1.0 / self.M_quad* self.volume * jnp.sum(((eval_impl_euler @ p_update).T + rhs_eval_l2_interior) ** 2)
                #     d += jnp.sum(p_update** 2)  * self.reg_eps ** 2 / tau ** 2
                #     d += jnp.sum((params - params0 + p_update)**2) * 1 / 2 * self.reg_eps ** 2 / tau ** 2
                #     
                ##     missing H1 stuff....
                #     
                #     return 0
                # dsq = lax.cond(i == self.gauss_iter_steps-1, lambda _: compute_dsq(), lambda _: dsq, None)
                return params + p_update, dsq

        '''
            for k in range(self.gauss_iter_steps):
                diff_quot = 1 / tau * (self.forward(params, self.xx_quad) - self.forward(params0, self.xx_quad))
                # THIS DOESN'T NEED TO BE COMPUTED ON FIRST PASS
                eval_laplacian_stage = laplacian_x_forward(params, self.xx_quad).reshape(1, -1)
                rhs_eval_l2_interior = diff_quot - eval_laplacian_stage
                rhs_l2_interior = self.h_squared * (
                            (self.quad_weights_interior.T * rhs_eval_l2_interior) @ eval_impl_euler).T

                rhs_border_stage = rhs_H1_border_helper(eval_dparam_border, eval_dparam_dx_border, params)
                rhs_border = rhs_border_stage - rhs_border_0_lambda

                rhs_reg = 1 / 2.0 * self.reg_eps ** 2 / tau ** 2 * (params - params0)
                rhs = -rhs_l2_interior - rhs_border - rhs_reg
                p_update = jax.scipy.linalg.cho_solve((c, low), rhs)
                params = params + p_update

            #delta_sq = 0
            #params, delta_sq = lax.fori_loop(0, self.gauss_iter_steps, loop_body, (params, delta_sq))

            return params
        #return step
        return jit(step)
