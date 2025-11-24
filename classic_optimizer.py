from traceback import print_tb

from jax import numpy as jnp, grad, jit


def make_jit_update(forward, target_function, lr=0.1):
    def loss_fn(params, x):
        return jnp.mean((forward(params, x).reshape(-1) - target_function(x))**2)

    def update(params, x):
        grads = grad(loss_fn)(params, x)
        params = params - lr * grads
        return params

    return jit(update)

def train_model_classic(forward, params, x, target_function, epochs=10000):
    update = make_jit_update(forward, target_function)
    for epoch in range(epochs):
        params = update(params, x)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {jnp.mean((forward(params, x) - target_function(x)) ** 2):.6f}")
    return params