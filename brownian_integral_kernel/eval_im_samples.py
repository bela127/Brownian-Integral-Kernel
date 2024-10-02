import numpy as np
from os import scandir

apr="bik" #bik bk bnk
name="exp_200_4.0_10_1"
path= f"./eval/{apr}/{name}/"

hyper_param = name.split(sep="_") if len(name.split(sep="_")) == 5 else [*name.split(sep="_")[:2], "1.5", *name.split(sep="_")[2:]]

stop_time = float(hyper_param[3])

points_per_interval = int(hyper_param[1])
train_intervals = 100

number_of_train_points = train_intervals * points_per_interval
interval_time = stop_time/train_intervals

pred_T = np.load(path+"pred_T.npy")
pred_Y = np.load(path+"pred_Y.npy")
pred_Var = np.load(path+"pred_Var.npy")

Samp_T = np.load(path+"Samp_T.npy")
Samp_Ys = np.load(path+"Samp_Ys.npy")

GT_T = np.load(path+"GT_T.npy")
GT_Ys = np.load(path+"GT_Ys.npy")

print(f"{pred_T.shape=}")
print(f"{pred_Y.shape=}")
print(f"{pred_Var.shape=}")

print(f"{Samp_T.shape=}")
print(f"{Samp_Ys.shape=}")

print(f"{GT_T.shape=}")
print(f"{GT_Ys.shape=}")

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

GT_mean_t, GT_interval_t, GT_mean_y, GT_int_y = integral_data(GT_T, GT_Ys)

sample_error = []
for sample in Samp_Ys.T:
    mean_t, interval_t, mean_y, int_y = integral_data(Samp_T,sample)

    mae = np.sum(np.abs(GT_int_y - int_y)) / float(int_y.shape[0])
    sample_error.append(mae)

av_mae = np.average(sample_error)
std_mae = np.std(sample_error)
print(f"{av_mae*10000:.3f}")
print(f"{std_mae*10000:.3f}")