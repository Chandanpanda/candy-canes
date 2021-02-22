# Candy Canes

## About
This repository contains our submission to the competition [Santa 2020: The Candy Cane Contest](https://www.kaggle.com/c/santa-2020) hosted on [Kaggle](https://www.kaggle.com). The bot finished X-th out of X teams on he [final leaderboard](https://www.kaggle.com/c/santa-2020/leaderboard).

## Local simulations
The script [run.py](run.py) lets the bot play locally against other user-supplied bots or built-in agents. It also produces some graphical representations of the resulting episode and stores a replay in JSON format. This requires the `kaggle_environments` package, which can be installed with the command
```
pip install kaggle_environments
```

Alternatively, install all the required packages using the following command
```
pip install -r requirements.txt
```

The script [load.py](load.py) produces the same graphics given an episode replay in JSON format (which could be taken from the leaderboard or produced by run.py).


## The strategy
The rules of the game can be found [here](https://www.kaggle.com/c/santa-2020/overview/environment-rules). In short, there are 100 arms that, when pulled by a player, return a candy cane with an initial probability that is specific to that arm and unknown to the players. For 2000 turns, each player can select an arm to pull on, after which the player with the most candy canes wins. After each turn, each player is informed of both the result of their last pull and of the arm pulled on by the opponent (but not the result of the opponent's pull). Finally, the reward probability for each arm decays by 3% each time that arm is pulled.

### Learning from Opponent Actions
We tried to hardcode the logic needed to learn from opponent actions. This can work well against agents who are greedy with respect to their threshold estimates, but these hardcoded rules are effectively countered by a deceptive opponent.

1. If the opponent repeats a first-time action, assume the first time is a success.
2. If the opponent hasn't pulled a lever in a long time then, it is probably because the first time was a failure.

### Heuristics
Somewhat surpringly, we noticed that our agents that did not use the opponent's actions to decay their estimates performed better than our agents that did use the opponent's actions to decay their estimates. One explanation for this is that not decaying encourages the agent to exploit good machines that the opponent has found, this is especially important early in the game. On the other hand, we figured that having correct estimates of the thresholds are more important near the end of the game. We attempted to get the best of both worlds by averaging these two estimates (with/without opponent decay) with a weight that favors non-decayed estimates early in the game and decayed estimates later in the game.


...

### Overview of the code
...