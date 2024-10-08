import GPy
import numpy as np
from matplotlib import pyplot as plt
from brownian_integral_kernel.integral_kernel import IntegralBrown
from brownian_integral_kernel.utils import create_fig, save
import pandas as pd

nr_plot_points = 10000
plt_t_start = 0
plt_t_end = 1000
y_min = -0.5
y_max = 1.5

start_t = 500
#                      sec*min*hour*day
end_t = 4000
# sec
measure_t = 5.0

points_per_interval = 15
interval_time = measure_t*points_per_interval

train_intervals = int((end_t - start_t) / interval_time)
number_of_train_points = train_intervals*points_per_interval
stop_time= number_of_train_points * measure_t

base_path="./brownian_integral_kernel/HIPE"

datasets = [
    "ChipPress_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
    "ChipSaw_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
    "HighTemperatureOven_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
    "MainTerminal_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
    "PickAndPlaceUnit_PhaseCount_2_geq_2017-10-23_lt_2017-10-30",
    "ScreenPrinter_PhaseCount_2_geq_2017-10-23_lt_2017-10-30",
    "SolderingOven_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
    "VacuumOven_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
    "VacuumPump1_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
    "VacuumPump2_PhaseCount_2_geq_2017-10-23_lt_2017-10-30",
    "WashingMachine_PhaseCount_3_geq_2017-10-23_lt_2017-10-30",
]

def load_hipe(path, name, start, length):
    
    df = pd.read_csv(path+"/"+name+".csv")
    df_window = df[start:start+length]

    t = np.arange(0, stop_time, measure_t)
    lp = df_window["U23_V"][:number_of_train_points].values
    lp=lp-lp[0]
    
    return t, lp


t_org, y_org = load_hipe(base_path, datasets[1], start_t, end_t)

def integral_data(t_org, y_org):
    aggregate_t = np.reshape(t_org, (train_intervals, points_per_interval))
    mean_t = np.mean(aggregate_t, axis=1)

    start_t = aggregate_t[:, 0]
    end_t = aggregate_t[:, 0] + interval_time

    interval_t = np.concatenate((start_t[:,None], end_t[:,None]), axis=1)
    inter_time = interval_t[:,1] - interval_t[:,0]

    aggregate_y = np.reshape(y_org, (train_intervals, points_per_interval))
    mean_y = np.mean(aggregate_y, axis=1)

    int_y = np.sum(aggregate_y , axis=1) * inter_time / points_per_interval
    int_y_1 = np.sum(aggregate_y * inter_time[:,None] / points_per_interval , axis=1) 
    int_y_2 = (mean_y * inter_time)
    int_y_3 = (mean_y * interval_time)

    return mean_t, interval_t, mean_y, int_y_3

mean_t, interval_t, mean_y, int_y = integral_data(t_org, y_org)


def plot_integral(ax, gp: GPy.models.GPRegression, title='Estimated Model'):
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredCov = gp.predict_noiseless(Xpred.T)
    SE = np.sqrt(YpredCov)

    ax.plot(t_org, y_org, label="Non-Integrated GT Data", linewidth=0.5, color="gray")
    ax.scatter(mean_t, mean_y,label='Measurements', marker="x", color="blue",linewidth=0.75)
    #ax.plot(Xpred[0], Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred[0],Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred[0],Ypred-SE*1.96,'r:')
    ax.set_title(title)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('target-variable $y$')

    ax.set_xlim(plt_t_start, plt_t_end)
    ax.set_ylim(y_min,y_max)

    plt.legend()

def plot_brown(ax, gp: GPy.models.GPRegression, title='Estimated Model'):
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = Xtest[:-1][:,None]
    Ypred,YpredCov = gp.predict_noiseless(Xpred)
    SE = np.sqrt(YpredCov)

    ax.plot(t_org, y_org, label="Non-Integrated GT Data", linewidth=0.5, color="gray")
    ax.scatter(mean_t, mean_y,label='Measurements', marker="x", color="blue",linewidth=0.75)
    #ax.plot(Xpred, Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred,Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred,Ypred-SE*1.96,'r:')
    ax.set_title(title)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('target-variable $y$')

    ax.set_xlim(plt_t_start, plt_t_end)
    ax.set_ylim(y_min,y_max)

    plt.legend()
    

k = IntegralBrown(variance=0.1)
m_ib = GPy.models.GPRegression(interval_t, int_y[:,None], k, noise_var=0.0)

fig_uc, axs_uc = create_fig(subplots=(1,2), width="paper")

fig_c_b, ax_c_b = create_fig(subplots=(1,1), width="paper", fraction=0.5)
fig_c_ib, ax_c_ib = create_fig(subplots=(1,1), width="paper", fraction=0.5)

print(m_ib)
plot_integral(axs_uc[1], m_ib, "Uncalibrated Model: $Int\_Brown$")

m_ib.Gaussian_noise.variance.fix()
print(m_ib)


m_ib.optimize_restarts(num_restarts=3, max_iters=1000, messages=True, ipython_notebook=False, parallel=True)
plot_integral(ax_c_ib, m_ib, "Model: $Int\_Brown$")


k = GPy.kern.Brownian(variance=0.01)
m_b = GPy.models.GPRegression(mean_t[:,None], mean_y[:,None], k, noise_var=0.0)
m_b.Gaussian_noise.variance.fix()
print(m_b)
plot_brown(axs_uc[0], m_b, "Uncalibrated Model: $Brown$")

m_b.optimize_restarts(num_restarts=5, max_iters=1000, messages=True, ipython_notebook=False, parallel=True)
print(m_b)
plot_brown(ax_c_b, m_b, "Model: $Brown$")

save(fig=fig_c_b, name="hipe_Brown", path="./exp_figures")

save(fig=fig_c_ib, name="hipe_Int_Brown", path="./exp_figures")


def sample_data(gp: GPy.models.GPRegression, size=5):

    t = np.linspace(0, stop_time, number_of_train_points*2)
    Xnew = np.concatenate((t[:,None], t[:,None]), axis=1)
    #For covariance calculation only second dimension is used, which should refer to the original time points

    print(gp)
    post: GPy.inference.latent_function_inference.exact_gaussian_inference.Posterior  = gp.posterior
    kern = gp.kern
    pred_var=gp._predictive_variable
    Kx = kern.K(pred_var, Xnew)
    mu = np.dot(Kx.T, post.woodbury_vector)

    #Kxx = kern.K(Xnew) #This call would calculate integral covariance!
    #But we want the Brownian covariance of GT variable instead:
    Xpart = Xnew[:,:1]
    Kxx = kern.variance*np.where(np.sign(Xpart)==np.sign(Xpart.T),np.fmin(np.abs(Xpart),np.abs(Xpart.T)), 0.)

    from GPy.util.linalg import dtrtrs, tdot
    tmp = dtrtrs(post._woodbury_chol, Kx)[0]
    var = Kxx - tdot(tmp.T)

    samples = np.random.multivariate_normal(mu.flatten(), var, size).T[:, np.newaxis, :]

    t_org = Xnew[:, :1]
    samples = samples[:,0,:]
    return t_org, samples

def plot_int_samples(ax, gp: GPy.models.GPRegression, title='Sampled Data'):
    global mean_t, mean_y
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredCov = gp.predict_noiseless(Xpred.T)
    SE = np.sqrt(YpredCov)

    t_org, samples = sample_data(gp, 4)
    ax.scatter(mean_t, mean_y,label='Measurements', marker="x", color="blue",linewidth=0.75)
    #ax.plot(Xpred[0], Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred[0],Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred[0],Ypred-SE*1.96,'r:')
    ax.plot(t_org, samples, label="Sampled Data", linewidth=0.5)

    #for sample in samples.T:
    #    mean_t, interval_t, mean_y, int_y = integral_data(t_org, sample)
    #   ax.scatter(mean_t, mean_y, label="Int Data", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('target-variable $y$')

    ax.set_xlim(plt_t_start, plt_t_end)
    ax.set_ylim(y_min,y_max)

    plt.legend()

def plot_brown_samples(ax, gp: GPy.models.GPRegression, title='Sampled Data'):
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = Xtest[:-1][:,None]
    Ypred,YpredCov = gp.predict_noiseless(Xpred)
    SE = np.sqrt(YpredCov)

    t = np.linspace(0, stop_time, number_of_train_points*2)
    print(gp)
    samples = gp.posterior_samples(t[:,None], full_cov=True, size=4)

    ax.scatter(mean_t, mean_y,label='Measurements', marker="x", color="blue",linewidth=0.75)
    #ax.plot(Xpred, Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred,Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred,Ypred-SE*1.96,'r:')
    ax.plot(t[:, None], samples[:,0,:], label="Sampled Data", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('target-variable $y$')

    ax.set_xlim(plt_t_start, plt_t_end)
    ax.set_ylim(y_min,y_max)

    plt.legend()

fig_c_b, ax_c_b = create_fig(subplots=(1,1), width="paper", fraction=0.5)
fig_c_ib, ax_c_ib = create_fig(subplots=(1,1), width="paper", fraction=0.5)


plot_int_samples(ax_c_ib, m_ib, "Model: $Int\_Brown$")
plot_brown_samples(ax_c_b, m_b, "Model: $Brown$")

save(fig=fig_c_b, name="hipe_Brown_samples", path="./exp_figures")

save(fig=fig_c_ib, name="hipe_Int_Brown_samples", path="./exp_figures")


#fig_c.tight_layout()
#plt.show()

