# Candy Canes

### About
This repository contains our submission to the competition [Santa 2020: The Candy Cane Contest](https://www.kaggle.com/c/santa-2020) hosted on [Kaggle](https://www.kaggle.com). The bot finished X-th out of X teams on he [final leaderboard](https://www.kaggle.com/c/santa-2020/leaderboard).

### Local simulations
The script [run.py](run.py) lets the bot play locally against other user-supplied bots or built-in agents. It also produces some graphical representations of the resulting episode and stores a replay in JSON format. This requires the `kaggle_environments` package, which can be installed with the command
```
pip install kaggle_environments
```
The script [load.py](load.py) produces the same graphics given an episode replay in JSON format (which could be taken from the leaderboard or produced by run.py).

### The strategy
The rules of the game can be found [here](https://www.kaggle.com/c/santa-2020/overview/environment-rules). In short, there are 100 arms that, when pulled by a player, return a candy cane with an initial probability that is specific to that arm and unknown to the players. For 2000 turns, each player can select an arm to pull on, after which the player with the most candy canes wins. After each turn, each player is informed of both the result of their last pull and of the arm pulled on by the opponent (but not the result of the opponent's pull). Finally, the reward probability for each arm decays by 3% each time that arm is pulled.

...

### Overview of the code
...