#!/usr/bin/python

import os
import random
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from kaggle_environments import make


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def eval_files(params):

    # unpack input
    seed, agent0, agent1 = params
    seed_everything(seed)

    # run the match
    env = make("mab", debug=True)
    env.run([agent0, agent1])
    
    # get the number of steps in the env
    steps = env.configuration.episodeSteps

    # observations at each step
    my_obs = [env.steps[s][0].observation for s in range(steps)]
    op_obs = [env.steps[s][1].observation for s in range(steps)]
    thresholds = [env.steps[s][0].observation.thresholds for s in range(steps)]
    thresholds = np.ceil(thresholds) / 100

    # extract raw scores
    scores0 = my_obs[-1].reward
    scores1 = op_obs[-1].reward

    # calculate expected scores
    escore0 = 0
    escore1 = 0
    for s in range(steps):
        action0 = env.steps[s][0].action
        escore0 += thresholds[s][action0]

        action1 = env.steps[s][1].action
        escore1 += thresholds[s][action1]

    # return results
    return dict(
        expected_score_0    = escore0,
        expected_score_1    = escore1,
        expected_win_rate_1 = 100*(escore1 > escore0),
        raw_score_0         = scores0,
        raw_score_1         = scores1,        
        raw_win_rate_1      = 100*(scores1 > scores0)
    )


def main():

    # parse user input
    parser = argparse.ArgumentParser(description="Evaluates matches with agent0 vs agent1.")

    parser.add_argument("agent0", type=str, 
                        help="Path to the file for agent0.") 

    parser.add_argument("agent1", type=str, 
                        help="Path to the file for agent1.") 
                        
    parser.add_argument("-n", "--n_threads", type=int, default=-1,
                        help="Number of cores to use when running matches.")

    parser.add_argument("-m", "--n_matches", type=int, default=100,
                        help="Number of matches to run.")

    args = parser.parse_args()

    print(f"Running {args.n_matches} matches with {args.agent0} vs {args.agent1}")
    if args.n_threads <= 0:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
    else:
        pool = multiprocessing.Pool(args.n_threads)

    results = pool.map(eval_files, [
        (seed, args.agent0, args.agent1)
        for seed in range(args.n_matches)
    ])

    pool.close()

    # aggregate results
    df = pd.DataFrame(results).mean()
    print("\n".join([f" * {x:<20} {y:>7.2f}" for x,y in zip(df.index, df.values)]))
    
    return


if __name__ == "__main__":
    main()
