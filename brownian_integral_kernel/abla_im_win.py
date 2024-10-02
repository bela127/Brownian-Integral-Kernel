from matplotlib import pyplot as plt
import numpy as np

from brownian_integral_kernel.utils import create_fig, save

w_sizes = np.asarray([25, 50, 100, 200])
results = np.asarray([7.6, 3.67, 2.0, 1.12])
vars = np.asarray([0.45, .26, .14, .09])


fig, ax = create_fig(subplots=(1,1), width="paper", fraction=0.5, hfrac=0.8)

ax.plot(w_sizes, results, 'r-', label=r'$\text{MAE}_{int}$')
ax.fill_between(w_sizes, results-vars*1.96, results+vars*1.96, color="blue", alpha=0.2, label=r'95%CI of $\text{MAE}_{int}$')

ax.set_title("Integral Error over Integral Window Size")
ax.set_xlabel('$w$')
ax.set_ylabel(r'$\text{MAE}_{int}$')

#ax.set_xlim(0, stop_time*1.2)
#ax.set_ylim(-3,3)

#plt.show()
plt.legend()

save(fig=fig, name="integral_error_over_w", path="./exp_figures")#, format="png")