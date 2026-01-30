import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

alpha = 0
data = loadmat(f"error_mats/errors_heat_lambda0_inicond_hs6_3e-5_alpha{alpha}_trapezoid_19_N_4-15_Eps_0.01-0.001-0.0001.mat")
errors = np.squeeze(data["errors"])

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

start_offset=0
iters = np.arange(4+start_offset, 15)

# Reference
e0 = errors[0][start_offset]

h = 2 ** iters
h0 = h[0]

ref_Oh   = e0 * (h0 / h)        # O(h)

plt.figure(figsize=(8, 5))
markers=['o-', 's-', 'x-', 'v-']
EPS = [0.01, 0.001, 0.0001]
for i, e in enumerate(EPS):
    plt.loglog(1/h, errors[i][start_offset:], markers[i], label=rf"$\epsilon = {e}$")

plt.loglog(1/h,ref_Oh, '--', label=r"$O(h)$")

plt.axhline(3.255605877701854e-05, linestyle='-.', linewidth=1,
            label=r"Initial condition $L^2$ error")

plt.xlabel("step size h", fontsize=16)
plt.ylabel(r"$L^2$-error at time $T=1$",fontsize=16)
plt.title(rf"Time convergence error, $\alpha={alpha}$", fontsize=20)
plt.legend()
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.savefig(f"plots/eps/heat_lambda0_inicond_hs6_3e-5_alpha{alpha}_trapezoid.eps", format="eps")
plt.show()
