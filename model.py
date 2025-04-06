"""Defines a specific markov chain by its parameters.

The model is defined by three sets of probabilities:
- iniital state probabilities
- state transition probabilities
- emission probabilities of observed states for each hidden state

The number of hidden and observed states are implicitly specified through the
dimensions of the probability matrices.
"""


import numpy as np


# Tnitial state probabilities
P0 = np.asarray([0.6, 0.1, 0.1, 0.2]).astype(np.float64)


# State transition probabilities
# P[i][j] is the i->j transition probability
P = np.asarray([
    [0.1, 0.4, 0.3, 0.2],
    [0.2, 0.3, 0.4, 0.1],
    [0.2, 0.2, 0.2, 0.4],
    [0.4, 0.2, 0.2, 0.2]
]).astype(np.float64)


# Emission probabilities
# O[i][j] is the probability of state i emitting output j
O = np.asarray([
    [0.2, 0.1, 0. , 0.7],
    [0.1, 0.1, 0.5, 0.3],
    [0.3, 0.4, 0.2, 0.1],
    [0.4, 0.1, 0.3, 0.2]
]).astype(np.float64)

