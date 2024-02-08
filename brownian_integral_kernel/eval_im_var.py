import numpy as np
from matplotlib import pyplot as plt


apr="bik" #bik bk bnk, rbfik
name="exp_25_10_1"
path= f"./eval/{apr}/{name}/"

stop_time = 10

points_per_interval = 25
train_intervals = 100

number_of_train_points = train_intervals * points_per_interval
interval_time = stop_time/train_intervals

pred_T = np.load(path+"pred_T.npy")
pred_Y = np.load(path+"pred_Y.npy")
pred_Var = np.load(path+"pred_Var.npy")

GT_T = np.load(path+"GT_T.npy")
GT_Ys = np.load(path+"GT_Ys.npy")

print(f"{pred_T.shape=}")
print(f"{pred_Y.shape=}")
print(f"{pred_Var.shape=}")

print(f"{GT_T.shape=}")
print(f"{GT_Ys.shape=}")

def std_data(y_org, pred_var):
    aggregate_y = np.reshape(y_org, (train_intervals, points_per_interval))
    org_var = np.var(aggregate_y, axis=1)

    avr_pred_var = pred_var[:,0][points_per_interval//2::points_per_interval]

    return np.sqrt(org_var), np.sqrt(avr_pred_var)

org_std, pred_std = std_data(GT_Ys, pred_Var)

#plt.plot(GT_T, np.sqrt(pred_Var[:,0]))
#plt.scatter(GT_T[7::15], np.ones_like(GT_T[7::15])*1.0e-5,c="red")
#plt.plot(GT_T[7::15], org_std)
#plt.plot(GT_T[7::15], pred_std)
#plt.show()

se = (org_std - pred_std)**2

mae = np.average(se)
std_ae = np.std(se)
print(f"{mae*1000:.2f}")
print(f"{std_ae*1000:.1f}")