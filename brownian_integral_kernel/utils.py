from dataclasses import dataclass
from typing import Any
from typing_extensions import Protocol
import matplotlib
from matplotlib import pyplot as plt
from nptyping import NDArray, Shape, Number, Float
import numpy as np
import os
import random


sim_time = 1000
base_path = "/home/bela/Cloud/Arbeit/KIT/Planed_Paper/estimation under brownean drift/experiment_raw_data/eval"
save_path = "/home/bela/Cloud/Arbeit/KIT/Planed_Paper/estimation under brownean drift/fig/exp_figures"

@dataclass
class RunRes():
    run_path: str
    run_name: str

    time: NDArray[Shape["Batch, 1"], Number]
    var: NDArray[Shape["Batch, 1"], Number]

    query: NDArray[Shape["Batch, 1"], Number]
    result: NDArray[Shape["Batch, 1"], Number]

    estimation: NDArray[Shape["Batch, 1"], Number]
    est_var: NDArray[Shape["Batch, 1"], Number]
    gt: NDArray[Shape["Batch, 1"], Number]

    query_time: NDArray[Shape["Batch, 1"], Number]
    query_var: NDArray[Shape["Batch, 1"], Number]   

def plot_pred_vs_gt(sim_res: RunRes):
    plt.plot(sim_res.time, sim_res.estimation)
    plt.plot(sim_res.time, sim_res.gt)

def calc_measurements(sim_res: RunRes)-> int:
    measurements: int = sim_res.result.shape[0]
    return measurements

def calc_nr_meas(sim_res: RunRes)-> NDArray[Shape["Batch, 1"], Number]:
    measurements = np.arange(1,sim_res.result.shape[0]+1)
    return measurements

def calc_meas_time(sim_res: RunRes) -> NDArray[Shape["Batch, 1"], Number]:
    time = sim_res.query[:,0]
    return time

def calc_rmse(sim_res: RunRes)-> float:
    rmse = np.sqrt(np.sum((sim_res.gt - sim_res.estimation)**2)/sim_res.gt.shape[0])
    return rmse

def calc_mean_std(datas):

    mean = np.mean(datas)
    std = np.std(datas)

    return mean, std

class FileWorker(Protocol):
    def __call__(self, file_path: str, file_name: str) -> Any: ...

def walk_files(path: str, worker: FileWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npy"):
                yield worker(file_path = dirpath, file_name = f)

class RunWorker(Protocol):
    def __call__(self, run_path: str, run_name: str) -> Any: ...

def walk_runs(path: str, worker: RunWorker):
    for dirpath, dnames, fnames in os.walk(path):
        d: str
        for d in dnames:
            if d.startswith("exp_"):
                yield worker(run_path = dirpath, run_name = d)

class SubExpWorker(Protocol):
    def __call__(self, sub_exp_path: str, sub_exp_name: str) -> Any: ...

def walk_dirs(path: str, worker: SubExpWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for d in dnames:
                yield worker(sub_exp_path = dirpath, sub_exp_name = d)

@dataclass
class EvalRes():
    exp_quantity: NDArray[Shape["Batch, 1"], Float]
    mean_rmse: NDArray[Shape["Batch, 1"], Float]
    std_rmse: NDArray[Shape["Batch, 1"], Float]
    mean_meas: NDArray[Shape["Batch, 1"], Float]
    std_meas: NDArray[Shape["Batch, 1"], Float]

def norm_time(time):
    return time / sim_time

def plot_rmse_over_exp_quant(ax, eval_res: EvalRes, exp_quant_name = "exp_quant", color=None):
    
    ax.plot(eval_res.exp_quantity, eval_res.mean_rmse, label=f"Mean {exp_quant_name}", color=color)
    ax.fill_between(
        eval_res.exp_quantity,
        eval_res.mean_rmse - eval_res.std_rmse*1.96,
        eval_res.mean_rmse + eval_res.std_rmse*1.96,
        alpha=0.1,
        color=color,
        #label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel(exp_quant_name)
    ax.set_ylabel("$RMSE$")

def plot_rmse_over_stdsel(ax, eval_res: EvalRes, exp_quant_name = "exp_quant", color=None):
    
    ax.plot(eval_res.exp_quantity, eval_res.mean_rmse, label=f"{exp_quant_name}", color=color)
    ax.fill_between(
        eval_res.exp_quantity,
        eval_res.mean_rmse - eval_res.std_rmse*1.96,
        eval_res.mean_rmse + eval_res.std_rmse*1.96,
        alpha=0.1,
        color=color,
        #label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel(r"$std_{\textit{target}} = \sqrt{v_{target}}$")
    ax.set_ylabel("$RMSE$")

def plot_rmse_over_mean_meas(ax, eval_res: EvalRes, exp_quant_name = "exp_quant", print_var =False, color=None, style=None, marker=None):

    if marker is not None:
        ax.plot(eval_res.mean_meas, eval_res.mean_rmse, color=color, linestyle=style)

        x,y = random.choice(list(zip(eval_res.mean_meas, eval_res.mean_rmse)))
        ax.scatter(x, y, marker=marker, label=exp_quant_name, color=color)
    else:
        ax.plot(eval_res.mean_meas, eval_res.mean_rmse, label=exp_quant_name, color=color, linestyle=style)

    if print_var:
        ax.fill_between(
            eval_res.mean_meas,
            eval_res.mean_rmse - eval_res.std_rmse*1.96,
            eval_res.mean_rmse + eval_res.std_rmse*1.96,
            alpha=0.2,
            color=color,
            label=r"$\pm$ 1 std. dev.",
            linewidth=0,
        )
    else:
        ax.fill_between(
            eval_res.mean_meas,
            eval_res.mean_rmse - eval_res.std_rmse*1.96,
            eval_res.mean_rmse + eval_res.std_rmse*1.96,
            alpha=0.2,
            color=color,
            linewidth=0,
        )

    ax.set_xlabel("Nr. of measurements")
    ax.set_ylabel('$RMSE$')

def plot_sum_meas_over_time(ax, meas_res, exp_quant_name = "exp_quant", color=None):

    for exp_quantity, times, mean_measures, std_meas in zip(*meas_res):
    
        ax.plot(times, mean_measures, label=f"Mean for {exp_quant_name}={exp_quantity:.3f}", color=color)
        ax.fill_between(
            times,
            mean_measures - std_meas*1.96,
            mean_measures + std_meas*1.96,
            alpha=0.1,
            color=color,
            #label=r"$\pm$ 1 std. dev.",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("total acquired measurements")

def plot_meas_over_time(ax, meas_res, exp_quant_name = "exp_quant", color=None):
    colors = []

    for i, (exp_quantity, times, mean_measures, std_meas) in enumerate(zip(*meas_res)):
        if isinstance(color, list):
            color = color[i]
    
        ax.plot(times, mean_measures/times, label=f"{exp_quant_name}={exp_quantity:.3f}", color=color)
        colors.append(plt.gca().lines[-1].get_color())
        ax.fill_between(
            times,
            (mean_measures - std_meas*1.96)/times,
            (mean_measures + std_meas*1.96)/times,
            alpha=0.1,
            color=plt.gca().lines[-1].get_color(),
            #label=r"$\pm$ 1 std. dev.",
        )
    ax.set_xlabel("time $t$")
    ax.set_ylabel("$m_{su}$ = meas. / $su$")
    return colors


def plot_meas_per_step_vs_exp_quant(ax, meas_res, exp_quant_name = "exp_quant", color=None):
    exp_quantities, _,_,_ = meas_res

    mean_meass = []
    mean_stds = []
    for exp_quantity, times, mean_measures, std_meas in zip(*meas_res):
        mean_meas = np.median(mean_measures/times)
        mean_std = np.median(std_meas/times)
        mean_meass.append(mean_meas)
        mean_stds.append(mean_std)
        if mean_meas ==  1.0:
            print(mean_meas)

    #for exp_quantity, mean_meas in list(zip(exp_quantities, mean_meass))[::2]:
    #    print(exp_quantity, mean_meas)

    exp_quantities = np.asarray(exp_quantities)
    mean_meass = np.asarray(mean_meass)
    mean_stds = np.asarray(mean_stds)
    
    ax.plot(exp_quantities, mean_meass, label=f"{exp_quant_name}", color=color)
    ax.fill_between(
        exp_quantities,
        mean_meass - mean_stds*1.96,
        mean_meass + mean_stds*1.96,
        alpha=0.1,
        color=color,
        #label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel(r"$v_{target}$")
    ax.set_ylabel(r"$\bar{m}_{su} = \mathbf{avr}[$meas.$ / su]$")


linestyle = {
     'dashdot': 'dashdot',  # Same as '-.'
     'solid':                 (0, ()),
     #'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     #'long dash with offset': (5, (10, 3)),
     #'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     #'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     #'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
}

colors = [
            "#88CCEE",
            "#CC6677",
            "#DDCC77",
            "#117733",
            "#332288",
            "#AA4499",
            "#44AA99",
            "#999933",
            "#882255",
            "#661100",
            "#888888"
        ]

marker = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "*",
    "+",
    "x",
    "d",
]

appr_style = {
    "CM": (colors[5], linestyle['dashed'], marker[5]),
    "CI": (colors[7], linestyle['solid'], marker[0]),
    "CEm": (colors[2], linestyle['dotted'], marker[1]),
    "CEo": (colors[6], linestyle['dashdotted'], marker[2]),
    "CEw": (colors[3], linestyle['densely dashdotdotted'], marker[4]),
    "CAL": (colors[1], linestyle['dashdot'], marker[13]),
    "CALm": (colors[8], linestyle['densely dashdotted'], marker[0]),
    "BR": (colors[0], linestyle['dashdotdotted'], marker[11]),
    "IB": (colors[4], linestyle['densely dashed'], marker[10]),
    "BRt": (colors[9], linestyle['dashdotdotted'],marker[12]),
}


def save(fig, name, path, format="svg"):
    #plt.title(name)
    fig.tight_layout()
    loc = os.path.join(path,f"{name}.{format}")
    fig.savefig(loc, format=format, bbox_inches='tight', transparent="True", pad_inches=0)
    fig.clf()

def set_size(width="paper_2c", fraction:float=1, subplots=(1, 1), hfrac=1., vfrac=1.):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'paper_2c':
        width_pt = 252
    elif width == 'paper':
        width_pt = 516
    elif width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif isinstance(width, float):
        width_pt = width
    else:
        raise ValueError(f"{width=} is no known size")

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio: float = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * 1.2 * hfrac * (subplots[0] / subplots[1])

    fig_width_in = fig_width_in * vfrac #Partial width

    return (fig_width_in, fig_height_in)

def create_fig(width="paper_2c", fraction:float =1, subplots=(1, 1), hfrac:float=1, vfrac:float=1):
    plt.style.use('seaborn-v0_8-paper')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True, #True,
        "text.latex.preamble": r'\usepackage{amssymb}',
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    plt.rcParams.update(tex_fonts)


    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=set_size(width=width, fraction=fraction, subplots=subplots, hfrac=hfrac, vfrac=vfrac))
    #axs.set_prop_cycle(line_cycler)
    return fig, axs



@dataclass
class DataComputer:
    sort_index: int = 3
    skip_points: int = 0
    sub_exp_folder: str = "exp_sub_folder"
    base_path: str = "./eval/exp_folder"

    def load_file(self, file_path, file_name):
        file_data = np.load(os.path.join(file_path, file_name), allow_pickle=True)
        return self.comp_file(file_data)

    def comp_file(self, file_data):
        return file_data

    def load_run(self, run_path, run_name):
        file_path = os.path.join(run_path, run_name)
        file_process = "all_data_process.npy"
        file_result = "all_data_result.npy"
        file_stream = "all_data_stream.npy"
        file_estimation = "estimation_data.npy"
        file_gt = "gt_data.npy"

        data_process = self.load_file(file_path, file_process)
        data_result = self.load_file(file_path, file_result)
        data_stream = self.load_file(file_path, file_stream)
        data_estimation = self.load_file(file_path, file_estimation)
        data_gt = self.load_file(file_path, file_gt)

        if data_estimation.shape[1] > 1:
            estimation = data_estimation[self.skip_points:,:1]
            est_var = data_estimation[self.skip_points:,1:]
        else:
            estimation = data_estimation[self.skip_points:]
            est_var = np.zeros_like(estimation)

        run_res = RunRes(
            run_path=run_path,
            run_name=run_name,
            time = data_stream[self.skip_points:,:1],
            var = data_stream[self.skip_points:,1:],
            query = data_result[self.skip_points:,:-1],
            result = data_result[self.skip_points:,-1:],
            estimation = estimation,
            est_var = est_var,
            gt = data_gt[self.skip_points:,-1:],
            query_time = data_gt[self.skip_points:,:1],
            query_var= data_gt[self.skip_points:,1:2]
        )
        return self.comp_run(run_res)
    
    def check_run_res_ok(self, run_res: RunRes):
        return not np.any(run_res.estimation == np.nan)

    def comp_run_not_ok(self, run_res: RunRes):
        print("Corrupt Sim Data: ", run_res.run_path)
        print("Min = ", np.min(run_res.estimation))
        run_res.estimation[run_res.estimation < 0] = 0
        run_res.estimation[run_res.estimation < 0] = 0

    def comp_run(self, run_res: RunRes):
        if not self.check_run_res_ok(run_res):
            self.comp_run_not_ok(run_res)

        rmse = calc_rmse(run_res)
        nr_measurements = calc_measurements(run_res)
        return rmse, nr_measurements


    def load_sub_exp(self, sub_exp_path: str, sub_exp_name: str):
        if sub_exp_name.startswith(self.sub_exp_folder):
            exp_quantity_str = sub_exp_name.removeprefix(self.sub_exp_folder)
            if exp_quantity_str:
                exp_quantity = float(exp_quantity_str)
            else:
                exp_quantity = None

            run_data = [run_result for run_result in walk_runs(path=os.path.join(sub_exp_path, sub_exp_name), worker=self.load_run)]

            data = exp_quantity, run_data

            return self.comp_sub_exp(data)

    def comp_sub_exp(self, data):
        exp_quantity, rmse_meas = data

        rmse , meas = np.asarray(rmse_meas).T

        mean_rmse, std_rmse = calc_mean_std(rmse)

        mean_meas, std_meas = calc_mean_std(meas)

        quant_rmse_std_meas_std = (exp_quantity, mean_rmse, std_rmse, mean_meas, std_meas)
        
        return quant_rmse_std_meas_std

    def load_exp(self):
        data = [i_r_v for i_r_v in walk_dirs(path=self.base_path, worker=self.load_sub_exp) if i_r_v is not None]
        return self.comp_exp(data)

    def comp_exp(self, loaded_data):
        inter_rmse_std_meas_std = np.asarray(loaded_data)
        inter_rmse_std_meas_std = inter_rmse_std_meas_std[inter_rmse_std_meas_std[:, self.sort_index].argsort()]
        exp_quantity, mean_rmse, std_rmse, mean_meas, std_meas = inter_rmse_std_meas_std.T

        eval_res = EvalRes(exp_quantity=exp_quantity, mean_rmse=mean_rmse, std_rmse=std_rmse, mean_meas=mean_meas, std_meas=std_meas)
        return eval_res


@dataclass
class ScatterRes():
    exp_quantity: NDArray[Shape["Batch, 1"], Float]
    rmse: NDArray[Shape["RepXBatch, 1"], Float]
    meas: NDArray[Shape["RepXBatch, 1"], Float]


def scatter_rmse_over_meas(ax, scat_res: ScatterRes, exp_quant_name = "exp_quant", color=None, marker=None):

    ax.scatter(scat_res.meas, scat_res.rmse, marker=marker, label=exp_quant_name, color=color, alpha=0.7)
    ax.set_xlabel("Nr. of measurements $N_m$")
    ax.set_ylabel('$RMSE$')

@dataclass
class ScatterComputer:
    sort_index: int = 3
    skip_points: int = 0
    sub_exp_folder: str = "exp_sub_folder"
    base_path: str = "./eval/exp_folder"

    def load_file(self, file_path, file_name):
        file_data = np.load(os.path.join(file_path, file_name))
        return self.comp_file(file_data)

    def comp_file(self, file_data):
        return file_data

    def load_run(self, run_path, run_name):
        file_path = os.path.join(run_path, run_name)
        file_process = "all_data_process.npy"
        file_result = "all_data_result.npy"
        file_stream = "all_data_stream.npy"
        file_estimation = "estimation_data.npy"
        file_gt = "gt_data.npy"

        data_process = self.load_file(file_path, file_process)
        data_result = self.load_file(file_path, file_result)
        data_stream = self.load_file(file_path, file_stream)
        data_estimation = self.load_file(file_path, file_estimation)
        data_gt = self.load_file(file_path, file_gt)

        if data_estimation.shape[1] > 1:
            estimation = data_estimation[self.skip_points:,:1]
            est_var = data_estimation[self.skip_points:,1:]
        else:
            estimation = data_estimation[self.skip_points:]
            est_var = np.zeros_like(estimation)

        run_res = RunRes(
            run_path=run_path,
            run_name=run_name,
            time = data_stream[self.skip_points:,:1],
            var = data_stream[self.skip_points:,1:],
            query = data_result[self.skip_points:,:-1],
            result = data_result[self.skip_points:,-1:],
            estimation = estimation,
            est_var = est_var,
            gt = data_gt[self.skip_points:,-1:],
            query_time = data_gt[self.skip_points:,:1],
            query_var= data_gt[self.skip_points:,1:2]
        )
        return self.comp_run(run_res)
    
    def check_run_res_ok(self, run_res: RunRes):
        return not np.any(run_res.estimation == np.nan)

    def comp_run_not_ok(self, run_res: RunRes):
        print("Corrupt Sim Data: ", run_res.run_path)
        print("Min = ", np.min(run_res.estimation))
        run_res.estimation[run_res.estimation < 0] = 0
        run_res.estimation[run_res.estimation < 0] = 0

    def comp_run(self, run_res: RunRes):
        if not self.check_run_res_ok(run_res):
            self.comp_run_not_ok(run_res)

        rmse = calc_rmse(run_res)
        nr_measurements = calc_measurements(run_res)
        return rmse, nr_measurements


    def load_sub_exp(self, sub_exp_path: str, sub_exp_name: str):
        if sub_exp_name.startswith(self.sub_exp_folder):
            exp_quantity_str = sub_exp_name.removeprefix(self.sub_exp_folder)
            if exp_quantity_str:
                exp_quantity = float(exp_quantity_str)
            else:
                exp_quantity = None

            run_data = [run_result for run_result in walk_runs(path=os.path.join(sub_exp_path, sub_exp_name), worker=self.load_run)]

            data = exp_quantity, run_data

            return self.comp_sub_exp(data)

    def comp_sub_exp(self, data):
        exp_quantity, rmse_meas = data

        rmse , meas = np.asarray(rmse_meas).T

        quant_rmse_meas = (exp_quantity, rmse, meas)
        
        return quant_rmse_meas

    def load_exp(self):
        data = [i_r_v for i_r_v in walk_dirs(path=self.base_path, worker=self.load_sub_exp) if i_r_v is not None]
        return self.comp_exp(data)

    def comp_exp(self, loaded_data):
        exp_quant = np.empty((0,1))
        rmses = np.empty((0,1))
        meass = np.empty((0,1))
        for exp_quantity, rmse, meas in loaded_data:
            exp_quant = np.concatenate((exp_quant, np.asarray([[exp_quantity]])))
            rmses = np.concatenate((rmses, rmse[:, None]))
            meass = np.concatenate((meass, meas[:, None]))

        scat_res = ScatterRes(exp_quantity=exp_quant, rmse=rmses, meas=meass)
        return scat_res

@dataclass
class MeasComputer(DataComputer):
    sort_index: int = 0

    def comp_run(self, run_res: RunRes):
        if not self.check_run_res_ok(run_res):
            self.comp_run_not_ok(run_res)

        meas = calc_nr_meas(run_res)
        time = calc_meas_time(run_res)
        return meas, time


    def comp_sub_exp(self, data):
        exp_quantity, meas_time = data

        meass , times = zip(*meas_time)

        all_times = np.unique(np.concatenate(times))
        sorted_times = np.sort(all_times )

        nr_runs = np.zeros_like(sorted_times)
        added_measurements = np.zeros_like(sorted_times)
        squared_meas = np.zeros_like(sorted_times)
        
        for i, time in enumerate(sorted_times):

            for run_meas, run_time in meas_time:
                index = np.where(run_time >= time)

                try:
                    index = index[0][0]
                    nr_runs[i] = nr_runs[i] + 1
                    added_measurements[i] = added_measurements[i] + run_meas[index]
                    squared_meas[i] = squared_meas[i] + run_meas[index]**2
                except IndexError:
                    ... # skip it



        mean_measures = added_measurements / nr_runs
        std_meas = np.sqrt(squared_meas / nr_runs - (added_measurements / nr_runs)**2)
        return exp_quantity, sorted_times, mean_measures, std_meas


    def comp_exp(self, loaded_data):        
        exp_quantity, times, mean_measures, std_meas = zip(*sorted(loaded_data, key=lambda x: x[self.sort_index]))

        
        return exp_quantity, times, mean_measures, std_meas