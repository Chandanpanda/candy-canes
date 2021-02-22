import numpy as np

# random number generator
rng = np.random.default_rng(13)

# two belief distributions for the thresholds of each arm
# the first only incorporates decay in the threshold resulting from our
# pulls, whereas the second incorporates decay resulting from both our pulls
# and those of the opponent
supports = np.tile(np.arange(101.0), (2, 100, 1))
beliefs = np.full_like(supports, 1 / 101.0)

# game history
my_pulls = np.array([], dtype=np.int)
op_pulls = np.array([], dtype=np.int)
results = np.array([], dtype=np.int)


def get_estimates(kind):
    # get optimistic estimates of the thresholds without any tilting
    mean = np.sum(supports[kind] * beliefs[kind], axis=1, keepdims=True)
    var = np.sum(((supports[kind] - mean) ** 2) * beliefs[kind], axis=1)
    untilted_estimates = np.squeeze(mean) + np.sqrt(var)

    # we "tilt" the beliefs distributions by applying some temporary
    # bayesian updates. if the opponent pulls an arm more than twice in the
    # last 10 turns, we update the beliefs as if all of these pulls were
    # sucesses. however, we stop tilting the beliefs of any arms which we know
    # have a bad untilted estimate, so we stop following the opponent if
    # they pull bad arms
    window = 10
    hist = np.bincount(op_pulls[-window:], minlength=100).reshape(100, 1)
    tilt = np.where(hist >= 2, hist, 0)
    tilt[untilted_estimates <= 25] = 0
    tilted = (np.ceil(supports[kind]) ** tilt) * beliefs[kind]
    tilted = tilted / tilted.sum(axis=1, keepdims=True)

    # if the opponent has pulled an arm exactly once in the last 101 turns,
    # we temporarily update the belief distribution for that arm as if
    # the opponent's pull was a failure. we exclude the last arm pulled
    # by the opponent from this rule.
    window = 101
    hist = np.bincount(op_pulls[-window:], minlength=100).reshape(100, 1)
    tilt = np.where(hist == 1, 1, 0)
    tilt[op_pulls[-1]] = 0
    tilted = ((101.0 - np.ceil(supports[kind])) ** tilt) * tilted
    tilted = tilted / tilted.sum(axis=1, keepdims=True)

    # return optimistic estimate of threshold given the tilted beliefs
    mean = np.sum(supports[kind] * tilted, axis=1, keepdims=True)
    var = np.sum(((supports[kind] - mean) ** 2) * tilted, axis=1)
    estimates = np.squeeze(mean) + np.sqrt(var)
    return estimates


def update(step):
    global beliefs, supports
    my_pull = my_pulls[-1]
    op_pull = op_pulls[-1]
    result = results[-1]

    # bayesian update for the result of our pull
    likelihood = np.ceil(supports[:, my_pull])
    likelihood = (101.0 - likelihood) if result == 0 else likelihood
    beliefs[:, my_pull] = likelihood * beliefs[:, my_pull]
    beliefs[:, my_pull] /= beliefs[:, my_pull].sum(axis=1, keepdims=True)

    # if the opponent repeats a first-time action,
    # assume the first pull was a success
    times_pulled = np.count_nonzero(op_pulls == op_pull)
    if (times_pulled == 2) and (op_pulls[-2] == op_pull):
        decay = np.array([[1], [0.97]])
        likelihood = np.ceil(supports[:, op_pull] / decay)
        beliefs[:, op_pull] = likelihood * beliefs[:, op_pull]
        beliefs[:, op_pull] /= beliefs[:, op_pull].sum(axis=1, keepdims=True)

    # if the opponent hasn't pulled a lever in a long time then
    # it is probably because the first time was a failure
    if step >= 102:
        pull = op_pulls[-102]
        if np.count_nonzero(op_pulls[:-1] == pull) == 1:
            decay = np.array([[1], [0.97]])
            likelihood = 101 - np.ceil(supports[:, pull] / decay)
            beliefs[:, pull] = likelihood * beliefs[:, pull]
            beliefs[:, pull] /= beliefs[:, pull].sum(axis=1, keepdims=True)

    # decay in thresholds due to the pulls. the decay due to the opponent
    # pull is only recorded in the belief distributions of the second kind
    supports[:, my_pull] *= 0.97
    supports[1, op_pull] *= 0.97
    return


# the main function called by kaggle environment for each turn
def agent(observation, configuration):
    global my_pulls, op_pulls, results

    # choose a random arm on the first turn
    if observation.step == 0:
        return int(rng.integers(0, 100))

    # parse observation
    my_pull = observation.lastActions[observation.agentIndex]
    op_pull = observation.lastActions[1 - observation.agentIndex]
    result = observation.reward - results.sum()

    # update game history
    my_pulls = np.append(my_pulls, my_pull)
    op_pulls = np.append(op_pulls, op_pull)
    results = np.append(results, result)

    # update belief distributions
    update(observation.step)

    # compute a weighted average of the estimates of the first and second
    # kinds. the weight is chosen based on how often the arm in question
    # has been pulled by both players.
    counts = np.bincount(np.append(my_pulls, op_pulls), minlength=100)
    weights = np.fmin(counts, 150) / 150
    estimates = (1 - weights) * get_estimates(0) + weights * get_estimates(1)

    # return a randomly chosen arm with a maximal estimate
    maximal = np.flatnonzero(estimates == estimates.max())
    return int(rng.choice(maximal))
