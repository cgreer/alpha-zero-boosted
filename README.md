
# Alpha Zero Boosted

A "build to learn" implementation of the [Alpha
Zero](https://doi.org/10.1038/nature24270) algorithm written in Python
that uses the [LightGBM](https://github.com/microsoft/LightGBM) (a Gradient
Boosted Decision Tree or GBDT) ML model in place of a Deep Neural Network for
the value/policy functions.

A few environments (i.e., games) are implemented: Quoridor, Connect
Four, and Tic-Tac-Toe.

![](https://github.com/cgreer/alpha_zero_boosted/raw/master/images/quoridor_sc.png)

![](https://github.com/cgreer/alpha_zero_boosted/raw/master/images/mcts_consideration_sc.png)


# Running

## Play a game

```python3.7 play.py <environment> <species-generation> <species-generation> <time-to-move>```

```python3.7 play.py connect_four mcts_gbdt-1 human 5.0```


## Train a bot

```python3.7 train_bot.py <environment> <species> <num batches>```

```python3.7 train_bot.py connect_four mcts_gbdt 10```
