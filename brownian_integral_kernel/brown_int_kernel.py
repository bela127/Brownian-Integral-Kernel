import GPy
import numpy as np
from matplotlib import pyplot as plt

from brownian_integral_kernel.data_generator import BrownianProcessDataSource
import numpy as np
from matplotlib import pyplot as plt # type: ignore

from brownian_integral_kernel.integral_kernel import IntegralBrown

std = 0.02
stop_time = 20

nr_plot_points = 10000

points_per_interval = 100
train_intervals = 15

number_of_train_points = train_intervals * points_per_interval
interval_time = stop_time/train_intervals


def compute_data():
    t = np.linspace(0, stop_time, number_of_train_points)

    ds = BrownianProcessDataSource(max_support=(stop_time,), support_points=2000, brown_var= std)()

    t, y = ds.query(t)

    return t, y

t_org, y_org = compute_data()

def integral_data(t_org, y_org):
    aggregate_t = np.reshape(t_org, (-1, points_per_interval))
    mean_t = np.mean(aggregate_t, axis=1)

    start_t = aggregate_t[:, 0]
    end_t = aggregate_t[:, 0] + interval_time

    interval_t = np.concatenate((start_t[:,None], end_t[:,None]), axis=1)

    aggregate_y = np.reshape(y_org, (-1, points_per_interval))
    int_y = np.sum(aggregate_y, axis=1) * interval_time
    mean_y = np.mean(aggregate_y, axis=1)
    return mean_t, interval_t, mean_y, int_y

mean_t, interval_t, mean_y, int_y = integral_data(t_org, y_org)


def plot_integral(gp: GPy.models.GPRegression):
    Xtest = np.linspace(0, stop_time*2, num=nr_plot_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredCov = gp.predict_noiseless(Xpred.T)
    SE = np.sqrt(YpredCov)

    plt.scatter(mean_t, mean_y)
    #plt.scatter(mean_t, int_y)
    plt.plot((Xpred[1]+Xpred[0])/2, Ypred,'r-',label='Mean')
    plt.plot((Xpred[1]+Xpred[0])/2,Ypred+SE*1.96,'r:',label='95% CI')
    plt.plot((Xpred[1]+Xpred[0])/2,Ypred-SE*1.96,'r:')
    plt.plot(t_org, y_org, label="Non-Integrated Data")
    plt.title('Estimated Model')
    plt.xlabel('time')
    plt.ylabel('y')
    plt.xlim(0, stop_time*1.2)
    #plt.ylim(-110,110)

res = (mean_y * (interval_t[:,1] - interval_t[:,0]))
k = IntegralBrown(variance=1)
m = GPy.models.GPRegression(interval_t, res[:,None], k, noise_var=0.0)
#train [100,2] ; 

print(m)
plot_integral(m)
plt.show()

m.Gaussian_noise.variance.fix()
print(m)


m.optimize_restarts(num_restarts=3, max_iters=1000, messages=True, ipython_notebook=False)
plot_integral(m)
plt.show()



