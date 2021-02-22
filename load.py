#!/usr/bin/python

import json
from box import Box, BoxKeyError

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # file containing replay in JSON format
    filename = "./episode.json"

    # load episode JSON
    with open(filename) as file:
        ep = Box(json.load(file))

    # get strings for the agent names
    try:
        agents = [n if n.isprintable() else "???" for n in ep.info.TeamNames]
    except BoxKeyError:
        agents = ["Player 0", "Player 1"]

    # get the number of steps in the episode
    steps = ep.configuration.episodeSteps

    # observations at each step
    my_obs = [ep.steps[s][0].observation for s in range(steps)]
    op_obs = [ep.steps[s][1].observation for s in range(steps)]
    thresholds = [ep.steps[s][0].observation.thresholds for s in range(steps)]

    # threshold of arms pulled compared to maximal threshold at each step
    my_thresholds = np.empty(steps)
    op_thresholds = np.empty(steps)
    optimal_thresholds = np.empty(steps)

    # expected scores
    my_escore = np.empty(steps)
    op_escore = np.empty(steps)

    # go through the episode and fill the arrays
    for s in range(steps):
        my_action = ep.steps[s][0].action
        my_thresholds[s] = thresholds[s][my_action]
        my_escore[s] = np.ceil(my_thresholds[s])

        op_action = ep.steps[s][1].action
        op_thresholds[s] = thresholds[s][op_action]
        op_escore[s] = np.ceil(op_thresholds[s])

        optimal_thresholds[s] = max(thresholds[s])

    my_escore = np.cumsum(my_escore) / 101
    op_escore = np.cumsum(op_escore) / 101

    # scatter plot of agent pulls
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x=np.arange(steps), y=my_thresholds,
                    label=agents[0], color="tab:blue", s=10)
    sns.lineplot(x=np.arange(steps), y=optimal_thresholds,
                 label="Optimal", color="tab:gray")
    plt.xlabel("Step")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    # scatter plot of opponent pulls
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x=np.arange(steps), y=op_thresholds,
                    label=agents[1], color="tab:orange", s=10)
    sns.lineplot(x=np.arange(steps), y=optimal_thresholds,
                 label="Optimal", color="tab:gray")
    plt.xlabel("Step")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    # scatter plot of all pulls
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x=np.arange(steps), y=op_thresholds,
                    label=agents[1], color="tab:orange", s=10)
    sns.scatterplot(x=np.arange(steps), y=my_thresholds,
                    label=agents[0], color="tab:blue", s=10)
    sns.lineplot(x=np.arange(steps), y=optimal_thresholds,
                 label="Optimal", color="tab:gray")
    plt.xlabel("Step")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    # line plots of expected scores
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
    print(f"\nMatch:    {agents[0]} -- {agents[1]}")
    print(f"Score:    {my_obs[-1].reward} -- {op_obs[-1].reward}")
    print(f"Expected: {round(my_escore[-1])} -- {round(op_escore[-1])}")

    return


if __name__ == "__main__":
    main()
