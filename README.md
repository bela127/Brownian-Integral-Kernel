# The Brownian Integral Kernel:
## A New Kernel for Modeling Integrated Brownian Motions

This is the repo containing code, data, and [supplementary material](https://github.com/bela127/Brownian-Integral-Kernel/blob/main/submat.pdf) for [the accompanying publication](https://github.com/bela127/Brownian-Integral-Kernel/blob/main/Paper.pdf).


## Installation:

install pyenv dependencies:

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
```

install pyenv:

```bash
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

add it to bash, by adding the following to the bottom of the file  `~/.bashrc`:

```bash
export PATH="/home/user/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

install python version:

```bash
pyenv install 3.9.6
```

install poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

add it to bash, by adding the following to the bottom of the file  `~/.bashrc`:

```bash
export PATH="/home/[user]/.local/bin:$PATH"
```

set poetry to use pyenv:

```bash
poetry config virtualenvs.prefer-active-python true
```

And make sure venv are created inside a project:

```bash
poetry config virtualenvs.in-project true
```

install project dependencies:

```bash
poetry install
```

wait for all dependencies to install and you are finished.

You can now simply import and use the 'IntegralBrown' as you would with any other GPy kernel:

```python
from brownian_integral_kernel.integral_kernel import IntegralBrown

... #Data loading

k = IntegralBrown(variance=1)
model = GPy.models.GPRegression(times, observation, k, noise_var=0.0)    
```

## Experiments and Evaluation

After the installation you can easily reproduce all experiments, evaluation and figures:
1. Simply run the "exp_..." scripts which will run the experiments on the datasets or on generated data.
2. Run the "eval_..." scripts to calculate the evaluation metrics.
3. Run the "vis_..." scripts to generate the plots seen in the paper.

And you are finished.

We also provide all load profile data in the "LP" subfolder.