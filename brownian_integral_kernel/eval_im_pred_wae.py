import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

apr="bik" #bik bk bnk rbfik
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

eps=0.0001
scale = np.sqrt(pred_Var[:,0])
scale[scale==0] = eps

max_like = norm.pdf(pred_Y[:,0], pred_Y[:,0], scale)

wae = max_like * np.abs(pred_Y[:,0] - GT_Ys)

#only use original measurement locations for calculation:
wae = wae[points_per_interval//2::points_per_interval]

av_wae = np.average(wae)
std_wae = np.std(wae)

print(f"{av_wae:.2f}")
print(f"{std_wae:.2f}")

