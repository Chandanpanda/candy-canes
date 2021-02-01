#!/usr/bin/python

import json
from box import Box
import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    epname = "badloss.json"

    # load episode JSON
    with open(os.path.join(os.curdir, "episodes", epname)) as file:
        ep = Box(json.load(file))

    # get strings for the agent names
    agents = [n if n.isprintable() else "???" for n in ep.info.TeamNames]

    # get the number of steps in the env
    steps = ep.configuration.episodeSteps

    # observations at each step
    my_obs = [ep.steps[s][0].observation for s in range(steps)]
    op_obs = [ep.steps[s][1].observation for s in range(steps)]
    thresholds = [ep.steps[s][0].observation.thresholds for s in range(steps)]

    # threshold of arms pulled compared to maximal threshold at each step
    my_threshold = np.empty(steps)
    op_threshold = np.empty(steps)
    optimal_threshold = np.empty(steps)

    # scores
    my_score = np.empty(steps)
    op_score = np.empty(steps)

    # expected score
    my_escore = np.empty(steps)
    op_escore = np.empty(steps)

    # go through the episode and fill the arrays
    for s in range(steps):
        my_action = ep.steps[s][0].action
        my_threshold[s] = thresholds[s][my_action]
        my_score[s] = my_obs[s].reward
        my_escore[s] = np.ceil(my_threshold[s])

        op_action = ep.steps[s][1].action
        op_threshold[s] = thresholds[s][op_action]
        op_score[s] = op_obs[s].reward
        op_escore[s] = np.ceil(op_threshold[s])

        optimal_threshold[s] = max(thresholds[s])

    my_escore = np.cumsum(my_escore) / 101
    op_escore = np.cumsum(op_escore) / 101

    # scatter plot of agent pulls
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x=np.arange(steps), y=my_threshold,
                    label=agents[0], color="tab:blue", s=10)
    sns.lineplot(x=np.arange(steps), y=optimal_threshold,
                 label="Optimal", color="tab:gray")
    plt.xlabel("Step")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    # scatter plot of opponent pulls
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x=np.arange(steps), y=op_threshold,
                    label=agents[1], color="tab:orange", s=10)
    sns.lineplot(x=np.arange(steps), y=optimal_threshold,
                 label="Optimal", color="tab:gray")
    plt.xlabel("Step")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    # scatter plot of opponent pulls
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x=np.arange(steps), y=op_threshold,
                    label=agents[1], color="tab:orange", s=10)
    sns.scatterplot(x=np.arange(steps), y=my_threshold,
                    label=agents[0], color="tab:blue", s=10)
    sns.lineplot(x=np.arange(steps), y=optimal_threshold,
                 label="Optimal", color="tab:gray")
    plt.xlabel("Step")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    # line plots of scores
    plt.figure(figsize=(15, 5))
    sns.lineplot(x=np.arange(steps), y=my_escore,
                 label=agents[0], color="tab:blue")
    sns.lineplot(x=np.arange(steps), y=op_escore,
                 label=agents[1], color="tab:orange")
    plt.xlabel("Step")
    plt.ylabel("Expected Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    # print scoreboard
    print(f"Match:    {agents[0]} -- {agents[1]}")
    print(f"Score:    {my_obs[-1].reward} -- {op_obs[-1].reward}")
    print(f"Expected: {round(my_escore[-1])} -- {round(op_escore[-1])}")

    return


if __name__ == "__main__":
    main()
