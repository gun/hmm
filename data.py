"""Functions to generate test data.
"""

import sys
import numpy as np
from numpy.typing import NDArray


def generate_seqs(
        P0: NDArray[np.float64],
        P: NDArray[np.float64],
        O: NDArray[np.float64],
        seqlen: int,
        count: int
) -> tuple[list, list, list]:
    """Generates a set of observed sequences given a model.

    Args:
        P0:     initial state probabilities
        P:      state transition matrix
        O:      emission probabilities
        seqlen: sequence length 
        count:  number of sequences

    Returns:
        A tuple as follows:

            (
              list of hidden sequences,
              list of corresponding observed sequences,
              observed sequences with the most likely observation state
            )
    """

    # Function to recursively generate a sequence of length k based on
    # specified transition probabilities   
    # NOTE: Not meant to be optimal
    states = np.arange(P0.shape[0])
    def generate(seq: list, k: int) -> list:
        if k == 0:
            return seq

        return generate(
            seq + [np.random.choice(states, p=P[seq[-1],:])],
            k-1
        )

    # Generate probabilistic hidden sequences
    seqs_hidden = []
    for i in range(count):
        sys.stdout.write(f"generating hidden sequences {i+1}\r")
        sys.stdout.flush()
        seqs_hidden.append(
            generate([np.random.choice(states, p=P0)], seqlen-1)
        )

    print("")

    # Generate corresponding observed sequences with probabilistic outputs
    print("generating observed sequences ...")
    seqs = [
        [np.random.choice(range(O.shape[1]), p=O[a,:]) for a in seq]
        for seq in seqs_hidden
    ]

    # Generate corresponding observed sequences with the highest probability
    # output
    print("generating most likely observed sequences ...")
    seqs_most_likely = [
        [np.argmax(O[a,:]) for a in seq]
        for seq in seqs_hidden
    ]

    return seqs_hidden, seqs, seqs_most_likely
