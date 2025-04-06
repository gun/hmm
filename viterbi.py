"""Tmplements the viterbi algorihtm for finding the most likely hidden state
sequence given an observed sequence corresponding to a specific hmm.
"""

import numpy as np
from numpy.typing import NDArray


def predict_state_seq(
        seq: list,
        P0: NDArray[np.float64],
        P: NDArray[np.float64],
        O: NDArray[np.float64]
):
    """ finds the most likely hidden state sequence given an observed sequence
    
    Args:
        seq:  observed sequence
        P0:   initial state probabilities
        P:    state transition matrix
        O:    emission probabilities

    Returns:
        The predicted hidden sequence as a list
    """

    # Initialize matrices:
    #
    # The score matrix keeps track of the probability of each prefix of the
    # given sequence to end with each of the possible hidden states
    #
    # The path matrix keeps track of the highest score path 
    matrix_score = np.zeros((len(seq), P0.shape[0]))
    matrix_path = np.zeros((len(seq), P0.shape[0]))
    
    # Build matrices
    for t in range(len(seq)):
        for i in range(P0.shape[0]):
            # Scores at t=0 are based on initial state probabilities rather
            # than transitions
            if t == 0:
                matrix_score[t,i] = P0[i] * O[i][seq[t]]
                continue

            scores = [matrix_score[t-1,j] * P[j][i] for j in range(P0.shape[0])]
            j = np.argmax(scores)

            matrix_score[t,i] = scores[j] * O[i][seq[t]]
            matrix_path[t,i] = j
            
        # Normalize to manage underflow
        matrix_score[t,:] = matrix_score[t,:] / np.max(matrix_score[t,:])

    # Decode to find the optimal path
    seq_decoded = []
    i = np.argmax(matrix_score[-1,:])
    for t in reversed(range(len(seq))):
        seq_decoded.append(int(i))
        i = matrix_path[t,int(i)]

    return list(reversed(seq_decoded))
