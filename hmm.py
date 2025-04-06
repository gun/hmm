"""core probabilistic functions that are used by various HMM algorithms.
"""


import numpy as np
from numpy.typing import NDArray


def alpha(
        seq: list,
        P0: NDArray[np.float64],
        P: NDArray[np.float64],
        O: NDArray[np.float64]
):
    """Implements a vectorized version of the alpha function for the forward
    algorithm for Hidden Markov Models.

    Args:
        seq:  observed sequence
        P0:   initial state probabilities
        P:    state transition matrix
        O:    emission probabilities

    Returns:
        A T-by-N matrix where T is the number of time steps in seq and N is the
        number of hidden states. each element (t, i) of the matrix indicates
        the probability of being in state i at time step t and having observed
        the sequence seq[0:t+1] until that time (where 0 <= t < T and
        0 <= i < N)
    """

    # Initialize matrix
    matrix = np.zeros((len(seq), P0.shape[0]))
    
    # Initial step
    matrix[0, :] = P0 * O[:,seq[0]]
    
    # Rest of the steps
    for t in range(1, len(seq)):
        matrix[t,:] = np.matmul(np.transpose(P), matrix[t-1,:]) * O[:,seq[t]]
 
    return matrix


def beta(
        seq: list,
        P0: NDArray[np.float64],
        P: NDArray[np.float64],
        O: NDArray[np.float64]
):
    """Implements a vectorized version of the beta function for the forward
    algorithm for Hidden Markov Models.

    Args:
        seq:  observed sequence
        P0:   initial state probabilities
        P:    state transition matrix
        O:   emission probabilities

    Returns:
        A T-by-N matrix where T is the number of time steps in seq and N is the
        number of hidden states. each element (t, i) of the matrix indicates
        the probability of being in state i at time step t and observing the
        sequence seq[t+1:T+1] after that time (where 0 <= t < T and 0 <= i < N)
    """

    # Initialize matrix
    matrix = np.zeros((len(seq), P0.shape[0]))
    
    # Initial step
    matrix[-1, :] = 1
    
    # Rest of the steps
    for t in reversed(range(len(seq)-1)):
        matrix[t,:] = np.matmul(P, matrix[t+1,:] * O[:,seq[t+1]])
 
    return matrix


def xi(
        seq: list,
        alpha_matrix: NDArray[np.float64],
        beta_matrix: NDArray[np.float64],
        A: NDArray[np.float64],
        B: NDArray[np.float64]
):
    """Implements a vectorized version of the xi function for the Baum-Welch
    algorithm.

    Args:
        seq:          observed sequence

        alpha_matrix: precomputed matrix of forward algorithm alpha values for
                      a given observed sequence. matrix should be T-by-N

        beta_matrix:  precomputed matrix of forward algorithm beta values for a
                      given observed sequence. matrix should be T-by-N

        A:            estimate of the hidden state transition probabilities.
                      N-by-N matrix where A[i][j] is the transition probability
                      from state i to j

        B:            estimate of the hidden state emission probabilities.
                      M-by-N matrix where B[i][j] is the emission probability of
                      observation j at state i

    Returns:
        A (T-1)-by-N-by-N array where T is the number of time steps and N is the
        number of hidden states. each N-by-N array at position t in the top level
        array represents the Baum-Welch xi(i,j) function at time=t.
    """

    matrix = []
    for t in range(len(seq) - 1):
        matrix_t = (
            alpha_matrix[t].reshape(-1, 1) *
            A *
            B[:, seq[t+1]] *
            beta_matrix[t+1]
        )
        matrix.append(matrix_t / np.sum(matrix_t))
        
    return np.asarray(matrix)


def gamma(
        alpha_matrix: NDArray[np.float64],
        beta_matrix: NDArray[np.float64]
):
    """Implements a vectorized version of the gamme function for the Baum-Welch
    algorithm.
    
    NOTE: This function is defined twice in the Introduction to Machine
    Learning (Alpaydin) book as eq.15.22 and eq.15.27. there's a subtle
    difference between them in that the latter does not provide values for the
    last time step in a sequence as it computes over xi values. the following
    implements 15.22

    Args:
        alpha_matrix:   precomputed matrix of forward algorithm alpha values
                        for a given observed sequence. matrix should be T-by-N

        beta_matrix:    precomputed matrix of forward algorithm beta values for
                        given observed sequence. matrix should be T-by-N

    Returns:
        A T-by-N matrix where T is the number of time steps and N is the number
        of hidden states. each value at position (t, i) represents the
        Baum-Welch gamma function value at time=t for state i for a given
        observed sequence
    """

    matrix = alpha_matrix * beta_matrix
    matrix = matrix / np.sum(matrix, axis=1).reshape(-1, 1)

    return matrix
