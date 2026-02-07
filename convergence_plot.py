import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

alpha = 0
quad = 10
quad_type = 'simpson'
lambda_damp = 0
T=0.75
data1 = loadmat(f"error_mats/errors_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_N_4-15_Eps_0.1-0.01-0.001-0.0001_T_{T}.mat")
#data2 = loadmat(f"error_mats/errors_heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_N_4-15_Eps_0.1-0.01.mat")
errors = data1["errors"]
#errors = np.vstack((data2["errors"], data1["errors"][0], data1["errors"][2]))

print(errors.shape)
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
EPS = [0.1, 0.01, 0.001, 0.0001]
for i in range(len(errors)):
    plt.loglog(1/h, errors[i][start_offset:], markers[i], label=rf"$\epsilon = {EPS[i]}$")

plt.loglog(1/h,ref_Oh, '--', label=r"$O(h)$")

plt.axhline(3.255605877701854e-05, linestyle='-.', linewidth=1,
            label=r"Initial condition $L^2$ error")

plt.xlabel("step size h", fontsize=16)
plt.ylabel(rf"$L^2$-error at time $T={T}$",fontsize=16)
if alpha == 0:
    s = rf"$L^2(\Gamma)$-norm"
elif alpha == 0.2:
    s = rf"weighted $H^1(\Gamma)$-norm"
elif alpha == 1:
    s = rf"$H^1(\Gamma)$-norm"
else:
    s = "SOME"
plt.title(rf"Time convergence error with "+s, fontsize=20)
plt.legend()
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.savefig(f"plots/eps/heat_lambda{lambda_damp}_inicond_hs6_3e-5_alpha{alpha}_{quad_type}_{quad}_T_{T}.eps", format="eps")
plt.show()
