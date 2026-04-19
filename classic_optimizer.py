from traceback import print_tb

from jax import numpy as jnp, grad, jit


def make_jit_update_real(forward, target_function, lr=0.1):
    def loss_fn(params, x):
        return jnp.mean((forward(params, x).reshape(-1) - target_function(x))**2)

    def update(params, x):
        grads = grad(loss_fn)(params, x)
        params = params - lr * grads
        return params

    return jit(update)

def make_jit_update_complex(forward, target_function, lr=0.1):

    def loss_fn(params, x):
        u = forward(params, x)[0] + 1j * forward(params, x)[1]
        im_noise = 1e-6j
        return jnp.mean(jnp.abs(u - target_function(x)+ im_noise)**2)

    def update(params, x):
        grads = grad(loss_fn)(params, x)
        params = params - lr * grads
        return params

    return jit(update)

def train_model_classic(forward, params, x, target_function, epochs=10000, complex=False):
    if not complex:
        update = make_jit_update_real(forward, target_function)
    else:
        update = make_jit_update_complex(forward, target_function)
    for epoch in range(epochs):
        params = update(params, x)
        if epoch % 2000 == 0:
            print(f"Epoch {epoch}, Loss: {jnp.mean(jnp.abs((forward(params, x) - target_function(x))) ** 2):.6f}")
    return params