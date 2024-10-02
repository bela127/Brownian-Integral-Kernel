from matplotlib import pyplot as plt
import numpy as np

from brownian_integral_kernel.utils import create_fig, save

data_var = np.asarray([0.5, 1.0, 1.5, 2.0, 4.0])

# w = 200
mae_200 = np.asarray([0.935, 0.999, 1.062, 1.343, 1.534])
mae_std_200 = np.asarray([0.074, 0.074, 0.072, 0.113, 0.091])

# w = 50
mae = np.asarray([2.090, 2.843, 3.834, 4.559, 5.998])
mae_std = np.asarray([0.115, 0.138, 0.273, 0.336, 0.376])



fig, ax = create_fig(subplots=(1,1), width="paper", fraction=0.5, hfrac=0.8)

ax.plot(data_var, mae, 'r-', label=r'$\text{MAE}_{int}$, $w=50$')
ax.fill_between(data_var, mae-mae_std*1.96, mae+mae_std*1.96, color="blue", alpha=0.2)

ax.plot(data_var, mae_200, 'g--', label=r'$\text{MAE}_{int}$, $w=200$')
ax.fill_between(data_var, mae_200-mae_std_200*1.96, mae_200+mae_std_200*1.96, color="blue", alpha=0.2, label=r'95%CI of $\text{MAE}_{int}$')

ax.set_title("Integral Error over Brownian Variance")
ax.set_xlabel('$v_b$')
ax.set_ylabel(r'$\text{MAE}_{int}$')

#ax.set_xlim(0, stop_time*1.2)
#ax.set_ylim(-3,3)

#plt.show()
plt.legend()

save(fig=fig, name="integral_error_over_var", path="./exp_figures")#, format="png")