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
export PATH="/home/i40/boehnkeb/.local/bin:$PATH"
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

wait for all dependencies to install.
