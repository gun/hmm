"""Test script that learns the parameters of a hmm given a data set of
sequences. 
"""


import sys
import copy

import numpy as np


from model import P0, P, O 
from data import generate_seqs
from em import EM


if __name__ == "__main__":
    # Generate  sequences following the given model
    seqlen = 50
    count = 100
    _, seqs, seqs_most_likely = generate_seqs(P0, P, O, seqlen, count)

    # Initialize expectation maximization
    model = EM(P0.shape[0], O.shape[1], seqs_most_likely)

    # Iterate until convergence
    # NOTE: Not actualling testing for convergence below
    for i in range(500):
        sys.stdout.write(f"iteration {i}\r")
        sys.stdout.flush()

        model.iterate()

        # Print the predicted initial state probabilities to monitor
        # convergence
        print("  ".join([f"{p:.4f}" for p in model.P0]))
