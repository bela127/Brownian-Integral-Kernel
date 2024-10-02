from os import makedirs, scandir
import GPy
import numpy as np
from matplotlib import pyplot as plt
from brownian_integral_kernel.integral_kernel import IntegralBrown
from brownian_integral_kernel.data_generator import BrownianProcessDataSource


nr_function_samples = 20

std = 1.0 #0.5, 1.0, 1.5, 2.0, 4.0
stop_time = 10

points_per_interval = 200
train_intervals = 100 

number_of_train_points = train_intervals * points_per_interval
interval_time = stop_time/train_intervals


def compute_data():
    t = np.linspace(0, stop_time, number_of_train_points)

    ds = BrownianProcessDataSource(max_support=(stop_time,), support_points=2000, brown_var= std)()

    t, y = ds.query(t)

    return t, y

def integral_data(t_org, y_org):
    aggregate_t = np.reshape(t_org, (train_intervals, points_per_interval))
    mean_t = np.mean(aggregate_t, axis=1)

    start_t = aggregate_t[:, 0]
    end_t = aggregate_t[:, 0] + interval_time

    interval_t = np.concatenate((start_t[:,None], end_t[:,None]), axis=1)
    #inter_time = interval_t[:,1] - interval_t[:,0]

    aggregate_y = np.reshape(y_org, (train_intervals, points_per_interval))
    mean_y = np.mean(aggregate_y, axis=1)

    #int_y = np.sum(aggregate_y , axis=1) * inter_time / points_per_interval
    #int_y_1 = np.sum(aggregate_y * inter_time[:,None] / points_per_interval , axis=1) 
    #int_y_2 = (mean_y * inter_time)
    int_y_3 = (mean_y * interval_time)

    return mean_t, interval_t, mean_y, int_y_3

def save_integral(gp: GPy.models.GPRegression, t_org, y_org, name, apr="bik"):
    Xtest = np.linspace(0, stop_time, num=number_of_train_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredVar = gp.predict_noiseless(Xpred.T)

    t_sam, samples = sample_data(gp)
    
    makedirs(f"./eval/{apr}/{name}", exist_ok=True)

    np.save(f"./eval/{apr}/{name}/pred_T.npy", arr=Xpred)
    np.save(f"./eval/{apr}/{name}/pred_Y.npy", arr=Ypred)
    np.save(f"./eval/{apr}/{name}/pred_Var.npy", arr=YpredVar)

    np.save(f"./eval/{apr}/{name}/Samp_T.npy", arr=t_sam[:,0])
    np.save(f"./eval/{apr}/{name}/Samp_Ys.npy", arr=samples)

    np.save(f"./eval/{apr}/{name}/GT_T.npy", arr=t_org)
    np.save(f"./eval/{apr}/{name}/GT_Ys.npy", arr=y_org)

def save_rbfik(gp: GPy.models.GPRegression, t_org, y_org, name, apr="rbfik"):
    Xtest = np.linspace(0, stop_time, num=number_of_train_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredVar = gp.predict_noiseless(Xpred.T)
    Ypred = - Ypred

    #t_sam = np.linspace(0, stop_time, number_of_train_points*2)
    #samples = gp.posterior_samples(t_sam[:,None], full_cov=True, size=nr_function_samples)
    # RPF Integral kernel does not provide a working sample function

    makedirs(f"./eval/{apr}/{name}", exist_ok=True)

    np.save(f"./eval/{apr}/{name}/pred_T.npy", arr=Xpred)
    np.save(f"./eval/{apr}/{name}/pred_Y.npy", arr=Ypred)
    np.save(f"./eval/{apr}/{name}/pred_Var.npy", arr=YpredVar)

    #np.save(f"./eval/{apr}/{name}/Samp_T.npy", arr=t_sam)
    #np.save(f"./eval/{apr}/{name}/Samp_Ys.npy", arr=samples)

    np.save(f"./eval/{apr}/{name}/GT_T.npy", arr=t_org)
    np.save(f"./eval/{apr}/{name}/GT_Ys.npy", arr=y_org)

def save_brown(gp: GPy.models.GPRegression, t_org, y_org, name, apr="bk"):
    Xtest = np.linspace(0, stop_time, num=number_of_train_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredVar = gp.predict_noiseless(Xpred.T)

    t_sam = np.linspace(0, stop_time, number_of_train_points)
    samples = gp.posterior_samples(t_sam[:,None], full_cov=True, size=nr_function_samples)

    makedirs(f"./eval/{apr}/{name}", exist_ok=True)
    
    np.save(f"./eval/{apr}/{name}/pred_T.npy", arr=Xpred)
    np.save(f"./eval/{apr}/{name}/pred_Y.npy", arr=Ypred)
    np.save(f"./eval/{apr}/{name}/pred_Var.npy", arr=YpredVar)

    np.save(f"./eval/{apr}/{name}/Samp_T.npy", arr=t_sam)
    np.save(f"./eval/{apr}/{name}/Samp_Ys.npy", arr=samples[:,0,:])

    np.save(f"./eval/{apr}/{name}/GT_T.npy", arr=t_org)
    np.save(f"./eval/{apr}/{name}/GT_Ys.npy", arr=y_org)


def sample_data(gp: GPy.models.GPRegression, size=nr_function_samples):

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


def experiment(name=f"exp_{points_per_interval}_{std:.1f}_{stop_time}_1"):
    t_org, y_org = compute_data()
    mean_t, interval_t, mean_y, int_y = integral_data(t_org, y_org)

    k = IntegralBrown(variance=std)
    m_ib = GPy.models.GPRegression(interval_t, int_y[:,None], k, noise_var=0.0)
    m_ib.Gaussian_noise.variance.fix()
    m_ib.optimize_restarts(num_restarts=3, max_iters=1000, messages=True, ipython_notebook=False, parallel=True)
    save_integral(m_ib, t_org, y_org, name)

#for i in range(20):
#    experiment(name=f"exp_{points_per_interval}_{std:.1f}_{stop_time}_{i}")

experiment()