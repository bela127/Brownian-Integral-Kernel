import os
import GPy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from brownian_integral_kernel.integral_kernel import IntegralBrown
from brownian_integral_kernel.utils import create_fig, save
from brownian_integral_kernel.data_generator import BrownianProcessDataSource

std = 1.5
stop_time = 10

nr_plot_points = 10000

points_per_interval = 110
train_intervals = 5

number_of_train_points = train_intervals * points_per_interval
interval_time = stop_time/train_intervals

sample_nr = 2

z = 100
y_min=-2
y_max=2

def compute_data():
    t = np.linspace(0, stop_time, number_of_train_points)

    ds = BrownianProcessDataSource(max_support=(stop_time,), support_points=2000, brown_var= std)()

    t, y = ds.query(t)

    return t, y

t_org, y_org = compute_data()

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
    ax.plot(Xpred[0], Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred[0],Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred[0],Ypred-SE*1.96,'r:')
    ax.set_title(title)
    ax.set_xlabel('time $t$')
    #ax.set_ylabel('target-variable $y$')

    ax.set_xlim(0, stop_time*1.2)
    #ax.set_xlim(0,10)
    ax.set_ylim(y_min,y_max)

def plot_brown(ax, gp: GPy.models.GPRegression, title='Estimated Model'):
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = Xtest[:-1][:,None]
    Ypred,YpredCov = gp.predict_noiseless(Xpred)
    SE = np.sqrt(YpredCov)

    ax.plot(t_org, y_org, label="Non-Integrated GT Data", linewidth=0.5, color="gray")
    ax.scatter(mean_t, mean_y,label='Measurements', marker="x", color="blue",linewidth=0.75)
    ax.plot(Xpred, Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred,Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred,Ypred-SE*1.96,'r:')
    ax.set_title(title)
    #ax.set_xlabel('time $t$')
    #ax.set_ylabel('target-variable $y$')

    ax.set_xlim(0, stop_time*1.2)
    #ax.set_xlim(0,10)
    ax.set_ylim(y_min,y_max)


def plot_gt(ax, title='Obscured Underlying Data'):
    ax.plot(t_org, y_org, label="Non-Integrated Underlying Data", linewidth=0.5, color="gray")
    ax.set_title(title)
    #ax.set_xlabel('time $t$')
    ax.set_ylabel('target $y$')

    ax.set_xlim(0, stop_time*1.2)
    ax.set_ylim(y_min,y_max)

def plot_measure(ax, title='Observed Aggregated Data'):
    ax.plot(t_org, y_org, label="Non-Integrated GT Data", linewidth=0.5, color="gray")
    ax.scatter(mean_t, mean_y,label='Integral Measurements', marker="x", color="blue",linewidth=0.75, zorder=z)
    ax.vlines(interval_t[:,1],np.ones_like(interval_t[:,1])*3,np.ones_like(interval_t[:,1])*-3)
    ax.set_title(title)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('target $y$')

    ax.set_xlim(0, stop_time*1.2)
    ax.set_ylim(y_min,y_max)
    

k = IntegralBrown(variance=1)
m_ib = GPy.models.GPRegression(interval_t, int_y[:,None], k, noise_var=0.0)

#fig_uc, axs_uc = create_fig(subplots=(1,2), width="paper")

#fig_c_b, ax_c_b = create_fig(subplots=(1,1), width="paper", fraction=0.5)
#fig_c_ib, ax_c_ib = create_fig(subplots=(1,1), width="paper", fraction=0.5)

#fig_c, axs_c = create_fig(subplots=(1,2), width="paper")

#ax_c_b = axs_c[0]
#ax_c_ib = axs_c[1]

print(m_ib)
#plot_integral(axs_uc[1], m_ib, "Uncalibrated Model: $Int\_Brown$")


m_ib.Gaussian_noise.variance.fix()
print(m_ib)


m_ib.optimize_restarts(num_restarts=3, max_iters=1000, messages=True, ipython_notebook=False)
#plot_integral(ax_c_ib, m_ib, "Calibrated Model: $Int\_Brown$")


k = GPy.kern.Brownian(variance=0.1)
m_b = GPy.models.GPRegression(mean_t[:,None], mean_y[:,None], k, noise_var=0.0)
m_b.Gaussian_noise.variance.fix()
print(m_b)
#plot_brown(axs_uc[0], m_b, "Uncalibrated Model: $Brown$")


m_b.optimize_restarts(num_restarts=5, max_iters=1000, messages=True, ipython_notebook=False)
print(m_b)
#plot_brown(ax_c_b, m_b, "Calibrated Model: $Brown$")


#save(fig=fig_c_b, name="Brown", path="./exp_figures")

#save(fig=fig_c_ib, name="Int_Brown", path="./exp_figures")


def sample_data(gp: GPy.models.GPRegression, size=5):

    t = np.linspace(0, stop_time, number_of_train_points)
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

    t_org, samples = sample_data(gp, sample_nr)
    #ax.plot(Xpred[0], Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred[0],Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred[0],Ypred-SE*1.96,'r:')
    ax.plot(t_org, samples, label="Sampled Data", linewidth=0.5)
    ax.scatter(mean_t, mean_y,label='Measurements', marker="x", color="blue",linewidth=0.75, zorder=z)


    #for sample in samples.T:
    #    mean_t, interval_t, mean_y, int_y = integral_data(t_org, sample)
    #   ax.scatter(mean_t, mean_y, label="Int Data", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel('time $t$')
    #ax.set_ylabel('target-variable $y$')

    ax.set_xlim(0, stop_time*1.2)
    ax.set_ylim(y_min,y_max)

def plot_brown_samples(ax, gp: GPy.models.GPRegression, title='Sampled Data'):
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = Xtest[:-1][:,None]
    Ypred,YpredCov = gp.predict_noiseless(Xpred)
    SE = np.sqrt(YpredCov)

    t = np.linspace(0, stop_time, number_of_train_points)

    print(gp)
    samples = gp.posterior_samples(t[:,None], full_cov=True, size=sample_nr)

    #ax.plot(Xpred, Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot(Xpred,Ypred+SE*1.96,'r:',label='$std_{est}$')
    ax.plot(Xpred,Ypred-SE*1.96,'r:')
    ax.plot(t_org[:, None], samples[:,0,:], label="Sampled Data", linewidth=0.5)
    ax.scatter(mean_t, mean_y,label='Measurements', marker="x", color="blue",linewidth=0.75, zorder=z)
    ax.set_title(title)
    #ax.set_xlabel('time $t$')
    #ax.set_ylabel('target-variable $y$')

    ax.set_xlim(0, stop_time*1.2)
    ax.set_ylim(y_min,y_max)

#fig_c_b, ax_c_b = create_fig(subplots=(1,1), width="paper", fraction=0.5)
#fig_c_ib, ax_c_ib = create_fig(subplots=(1,1), width="paper", fraction=0.5)

#fig_s, axs_s = create_fig(subplots=(1,2), width="paper")

#ax_s_b = axs_s[0]
#ax_s_ib = axs_s[1]

#plot_int_samples(ax_s_ib, m_ib)
#plot_brown_samples(ax_s_b, m_b)

fig_a, axs_a = create_fig(subplots=(2,3), width="paper",hfrac=1)

plot_gt(axs_a[0,0], title="a) Obscured Underlying Data")
plot_measure(axs_a[1,0], title="b) Observed Aggregated Data")

plot_brown(axs_a[0,1], m_b, "c) Conventional BK")
plot_integral(axs_a[1,1], m_ib, "d) Our BIK")


plot_brown_samples(axs_a[0,2], m_b, title="e) BK Data Synthesis")
plot_int_samples(axs_a[1,2], m_ib, title="f) BIK Data Synthesis")



#frame1.axes.xaxis.set_ticklabels([])
axs_a[1,1].yaxis.set_ticklabels([])
axs_a[0,1].yaxis.set_ticklabels([])
axs_a[1,2].yaxis.set_ticklabels([])
axs_a[0,2].yaxis.set_ticklabels([])


axs_a[0,0].xaxis.set_ticklabels([])
axs_a[0,1].xaxis.set_ticklabels([])
axs_a[0,2].xaxis.set_ticklabels([])


vl = lines.Line2D([], [], color='#1f77b4', marker='|', linestyle='None',
                          markersize=10, markeredgewidth=1.5, label='Integration Windows')

leg = fig_a.legend()
hands = leg.legend_handles

new_hs = [vl]
new_labels = [vl.get_label()]
use = [0,10,13,14]
for i,h in enumerate(hands):
    print(i, ":", h.get_label(), h)
    if i in use:
        new_hs.append(h)
        new_labels.append(h.get_label())

new_hs.append((hands[16],hands[17]))
new_labels.append(hands[16].get_label())

leg.remove()
leg = fig_a.legend(handles=new_hs, labels = new_labels ,loc="outside lower center", ncols=6, handler_map={tuple: HandlerTuple(ndivide=None)})

plt.tight_layout(rect=(0, 0.08, 0.99, 1),pad=0, h_pad=0.5, w_pad=0.6)

format = "svg"
name = "Intro_fig"
path = "./exp_figures"
loc = os.path.join(path,f"{name}.{format}")

fig_a.savefig(loc, format=format, bbox_inches='tight', transparent="True", pad_inches=0)

plt.show()

#save(fig=fig_c_ib, name="Int_Brown_samples", path="./exp_figures")


#fig_c.tight_layout()
#plt.show()

