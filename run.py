#!/usr/bin/python

import json
import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from kaggle_environments import make


def main():
    # specify what agents to match. can choose from built-in agents
    # "random" and "round_robin", provide a function, or a filename
    agents = ["./agent.py", "random"]

    # run the simulation
    env = make("mab", debug=True)
    env.run(agents)

    # save a record of the episode to episode.json
    with open(os.path.join(os.curdir, "episode.json"), "w") as file:
        json.dump(env.toJSON(), file)

    # get the number of steps in the episode
    steps = env.configuration.episodeSteps

    # observations at each step
    my_obs = [env.steps[s][0].observation for s in range(steps)]
    op_obs = [env.steps[s][1].observation for s in range(steps)]
    thresholds = [env.steps[s][0].observation.thresholds for s in range(steps)]

    # threshold of arms pulled compared to maximal threshold at each step
    my_thresholds = np.empty(steps)
    op_thresholds = np.empty(steps)
    optimal_thresholds = np.empty(steps)

    # expected scores
    my_escore = np.empty(steps)
    op_escore = np.empty(steps)

    # go through the episode and fill the arrays
    for s in range(steps):
        my_action = env.steps[s][0].action
        my_thresholds[s] = thresholds[s][my_action]
        my_escore[s] = np.ceil(my_thresholds[s])

        op_action = env.steps[s][1].action
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
