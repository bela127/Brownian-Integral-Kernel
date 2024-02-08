import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

apr="bik" #bik bk bnk rbfik
name="LP_H_2Erw_2Ki_wkB_mB_15-21_03_21_1"
path= f"./eval/{apr}/{name}/"

points_per_interval = 15


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
like = norm.pdf(GT_Ys, pred_Y[:,0], scale)

rel_like = like / max_like

#only use original measurement locations for calculation:
rel_like = rel_like[points_per_interval//2::points_per_interval]

av_like = np.average(rel_like)
std_like = np.std(rel_like)

print(f"{av_like:.2f}")
print(f"{std_like:.2f}")