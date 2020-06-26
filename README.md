
# Alpha Zero Boosted

A "build to learn" implementation of the [Alpha
Zero](https://doi.org/10.1038/nature24270) algorithm written in Python
that uses [LightGBM](https://github.com/microsoft/LightGBM) (Gradient
Boosted Decision Tree) in place of a Deep Neural Network for value/policy functions.

A few environments (i.e., games) are implemented: Quoridor, Connect
Four, and Tic-Tac-Toe.

![](https://github.com/cgreer/alpha_zero_boosted/raw/master/images/quoridor_sc.png)

![](https://github.com/cgreer/alpha_zero_boosted/raw/master/images/mcts_consideration_sc.png)

<br />


# Running

### Play a game

```python3.7 play.py <environment> <species-generation> <species-generation> <time-to-move>```

```python3.7 play.py connect_four gbdt-1 human-1 5.0```


### Train a bot

```python3.7 train_bot.py <environment> <species> <num batches>```

```python3.7 train_bot.py connect_four gbdt 10```

   
<br />

# Setup

<br />

### Installing

Install pyenv/pyenv-virtualenv (if you haven't already)
- See "Installing pyenv/pyenv-virutalenv" below

Clone repo

```git clone git@github.com:cgreer/alpha-zero-boosted.git```

Create virtual environment for project

```bash
cd alpha-zero-boosted
pyenv install 3.7.7
pyenv virtualenv 3.7.7 alpha_boosted_env
pyenv local alpha_boosted_env
```

Install packages

```pip install -r requirements.txt```

Because some wheels don't appear to be built properly, you may need to first install the following:

```bash
brew install libomp

# Then try installing again
pip install -r requirements.txt
```

<br />

### Installing pyenv/pyenv-virutalenv
        
**Installing pyenv**

```brew install pyenv```

**Install pyenv-virtualenv plugin**
Note:
- These instructions are copied here for convenience, check [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv#installing-with-homebrew-for-macos-users) to ensure they are up to date

Install the plugin:

```brew install pyenv-virtualenv```

Add these two lines to your profile file (~/.zprofile if using zsh, ~/.bash_profile if bash):

```bash
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Restart terminal (so profile commands above execute).


