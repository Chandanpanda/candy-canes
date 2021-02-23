# Candy Canes

## About
This repository contains our submission to the competition [Santa 2020: The Candy Cane Contest](https://www.kaggle.com/c/santa-2020) hosted on [Kaggle](https://www.kaggle.com). The bot finished 14th out of 792 teams on the [final leaderboard](https://www.kaggle.com/c/santa-2020/leaderboard). Our strategy relies on a Bayesian bandit algorithm with some additional heuristics.


## Local simulations
The script [run.py](run.py) lets the bot play locally against other user-supplied bots or built-in agents. It also produces some graphical representations of the resulting episode and stores a replay in JSON format. The script [load.py](load.py) produces the same graphics given an episode replay in JSON format (which could be taken from the leaderboard or produced by run.py). The required packages can be installed with the command
```
pip install -r requirements.txt
```

## The strategy
#### Rules of the game
The rules of the game can be found [here](https://www.kaggle.com/c/santa-2020/overview/environment-rules). In short, there are 100 arms that, when pulled by a player, return a candy cane with an initial probability that is specific to that arm and unknown to the players. For 2000 turns, each player can select an arm to pull on, after which the player with the most candy canes wins. After each turn, each player is informed of both the result of their last pull and of the arm pulled by the opponent (but not the result of the opponent's pull). Finally, the reward probability for each arm decays by 3% each time that arm is pulled.

#### Bayesian approach
The logic of [agent.py](agent.py) is based on maintaining a set of distributions reflecting what we know about the current reward probability of each arm. At the outset, these distributions are initialized to a uniform prior. Every time we get a result from our previous pull, we perform a Bayesian update on the corresponding distribution reflecting this new information. This is a [classical idea](https://en.wikipedia.org/wiki/Thompson_sampling) for dealing with multi-armed bandits.

#### Choosing the next arm
We decide on a new arm to pull using a simplified variant of the [Bayesian UCB](http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf) algorithm. We get an optimistic estimate of the reward probability for each arm by adding the mean and the standard deviation of the corresponding distribution and then choose an arm with a maximal estimate. An important part of the game is to understand when the opponent has discovered an arm with a high reward probability before the opponent has exploited that arm enough to dramatically decrease the probability. We tried to incorporate this information by applying some temporary Bayesian updates to the distribution before computing the mean and standard deviation comprising the estimate:
*  If the opponent pulled an arm more than twice in the last 10 turns, we assume all of those pulls resulted in a reward.
* If the opponent pulled an arm exactly once in the last 100 turns, we assume that pull did not result in a reward. We exclude the very last arm pulled by the opponent from this rule.

Our local simulations indicated that an optimal strategy never pulls on an arm with a reward probability less than ~0.2. Therefore, we stopped applying temporary updates to arms for which the untilted estimate (without any temporary updates) was less than 0.25. This ensures that we don't follow the opponent to arms with a bad probability when we already know that the reward probability is too small.

#### Decaying thresholds
In principle, the distributions should also reflect the decay in the reward probability after each arm is pulled. However, we noticed that our agents that did not use the opponent's actions to decay their estimates performed surprisingly well. One explanation for this is that not decaying encourages the agent to exploit good arms that the opponent has found - this is especially important early in the game. On the other hand, having correct estimates of the reward probabilities is important near the end of the game. We attempted to get the best of both worlds by averaging these two estimates (with/without opponent decay) with a weight that favors non-decayed estimates early in the game and decayed estimates later in the game.

#### Learning from opponent actions
Finally, some opponent actions are so indicative of the results of their previous pulls that we conservatively applied some permanent Bayesian updates to the distributions. We hardcoded the logic needed to learn from these opponent actions. This can work well against agents that rely heavily on the immediate exploitation of good arms, but these hardcoded rules are easy to counter by being deceptive. Here are the rules that we used:
* If the opponent repeats a first-time action, assume the first time is a success.
* If the opponent doesn't pull a lever for a long time after pulling it for the first time,  it is probably because the first time was a failure.
