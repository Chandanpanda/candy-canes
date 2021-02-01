import numpy as np

# random number generator
rng = np.random.default_rng(13)

# belief distribution for the thresholds of each arm
time_since_op_pull = np.zeros(100)
action_counts = np.zeros((100, 2), np.int16)
supports = np.tile(np.arange(101.0), (2, 100, 1))
beliefs = np.full_like(supports, 1 / 101.0)

# game history
total_reward = 0
my_pulls = np.array([], dtype=np.int)
op_pulls = np.array([], dtype=np.int)
results = np.array([], dtype=np.int)

def get_estimates(support_type):

    window = 10
    hist = np.bincount(op_pulls[-window:], minlength=100).reshape(100, 1)
    hist = np.where(hist >= 2.0, hist, 0.0)

    tilted = (np.ceil(supports[support_type]) ** hist) * beliefs[support_type]
    tilted = tilted / tilted.sum(axis=1, keepdims=True)

    hist = np.bincount(op_pulls[-101:], minlength=100).reshape(100, 1)
    hist = np.where(hist == 1, 1.0, 0.0)
    hist[op_pulls[-1]] = 0.0

    tilted = ((101.0 - np.ceil(supports[support_type])) ** hist) * tilted
    tilted = tilted / tilted.sum(axis=1, keepdims=True)
    
    # get optimistic estimate of threshold
    optimism = 1.0
    mean = np.sum(supports[support_type] * tilted, axis=1, keepdims=True)
    var = np.sum(((supports[support_type] - mean) ** 2) * tilted, axis=1, keepdims=True)
    estimates = np.squeeze(mean + optimism * np.sqrt(var))

    # return estimates
    return estimates


def update():
    global beliefs, supports, action_counts, time_since_op_pull
    my_pull = my_pulls[-1]
    op_pull = op_pulls[-1]
    result = results[-1]

    # bayesian update for the result of our pull
    likelihood = np.ceil(supports[:, my_pull])
    likelihood = (101.0 - likelihood) if result == 0 else likelihood
    beliefs[:, my_pull] = likelihood * beliefs[:, my_pull]
    beliefs[:, my_pull] /= beliefs[:, my_pull].sum(1, keepdims=True)

    # if the opponent repeats a first-time action, assume the first time is a success
    if (action_counts[op_pull,1] == 2) and (op_pulls[-2] == op_pull):
        likelihood = np.ceil(supports[:, op_pull]/np.array([[1], [0.97]]))
        beliefs[:, op_pull] = likelihood * beliefs[:, op_pull]
        beliefs[:, op_pull] /= beliefs[:, op_pull].sum(1, keepdims=True)

    # if the opponent hasn't pulled a lever in a long time then
    # it is probably because the first time was a failure
    for pull in np.where((action_counts[:,1] == 1) & (time_since_op_pull > 100))[0]:
        likelihood = 101 - np.ceil(supports[:, pull]/np.array([[1], [0.97]]))
        beliefs[:, pull] = likelihood * beliefs[:, pull]
        beliefs[:, pull] /= beliefs[:, pull].sum(1, keepdims=True)
        time_since_op_pull[pull] = -np.inf

    # decay in threshold due to pull
    supports[:, my_pull] *= 0.97

    # decay due to opponent's pull
    supports[1, op_pull] *= 0.97

    # increment counts
    action_counts[my_pull,0] += 1
    action_counts[op_pull,1] += 1

    # update time since pull
    time_since_op_pull[op_pull] = 0
    time_since_op_pull += 1
    return


# the main function called by kaggle environment for each turn
def agent(observation, configuration):
    global total_reward, action_counts, time_since_op_pull
    global my_pulls, op_pulls, results

    if observation['step'] == 0:
        return int(rng.integers(0, 100))

    my_pull = observation.lastActions[observation.agentIndex]
    op_pull = observation.lastActions[1 - observation.agentIndex]
    result = observation.reward - results.sum()

    # update game history
    my_pulls = np.append(my_pulls, my_pull)
    op_pulls = np.append(op_pulls, op_pull)
    results = np.append(results, result)
    update()

    # get action
    resistance = 250 # larger values make switching slower
    w = np.minimum(action_counts.sum(1), resistance) / resistance
    estimates = (1 - w)*get_estimates(0) + w*get_estimates(1) 
    maximums = np.flatnonzero(estimates == estimates.max())
    return int(rng.choice(maximums))
