import numpy as np

def load_profile(path):
    
    df = np.genfromtxt(path, delimiter= ';', skip_header=True)[:,2]
    #the data is given as total energy consumption during one minute in kWh.
    t = np.arange(0, df.shape[0], 1.0)
    #multiplying with 60 yields the power
    
    return t, df


t_org, y_org = load_profile("./brown/LP_H_2Erw_3Ki_wB_mB_15-21_03_21_3.csv")

start_t = 0
#                      min*hour*day
end_t = t_org[-1]
#
measure_t = 1.0

points_per_interval = 15
interval_time = measure_t*points_per_interval

train_intervals = int((end_t - start_t) / interval_time)
number_of_train_points = train_intervals*points_per_interval
stop_time= number_of_train_points * measure_t

t_org = t_org[:number_of_train_points]
y_org = y_org[:number_of_train_points]

def integral_data(t_org, y_org):
    aggregate_t = np.reshape(t_org, (train_intervals, points_per_interval))
    mean_t = np.mean(aggregate_t, axis=1)

    start_t = aggregate_t[:, 0]
    end_t = aggregate_t[:, 0] + interval_time

    interval_t = np.concatenate((start_t[:,None], end_t[:,None]), axis=1)
    inter_time = interval_t[:,1] - interval_t[:,0]

    aggregate_y = np.reshape(y_org, (train_intervals, points_per_interval))
    mean_y = np.mean(aggregate_y, axis=1)

    #This is mathematically all the same, but because of numerical errors the used method is the best (for fixed integration intervals)
    int_y = np.sum(aggregate_y , axis=1) * inter_time / points_per_interval
    int_y_1 = np.sum(aggregate_y * inter_time[:,None] / points_per_interval , axis=1) 
    int_y_2 = (mean_y * inter_time)
    int_y_3 = (mean_y * interval_time)

    return mean_t, interval_t, mean_y, int_y_3

mean_t, interval_t, mean_y, int_y = integral_data(t_org, y_org)

data = np.concatenate((interval_t, int_y[:,None]), axis=1)

np.savetxt("./integrated_data.csv", data, delimiter=";", header="Start Time; Stop Time; Value in kWh/15 minutes")