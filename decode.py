"""Test script that decodes observed sequences to predict hidden sequences.
"""


import sys
import numpy as np

from model import P0, P, O 
from data import generate_seqs
from viterbi import predict_state_seq


if __name__ == "__main__":
    # Generate a data set of observed sequences from the model.
    # NOTE: Two sets of sequences are generated. the first using the actual
    # emission probabilities of the model and the second by always picking the
    # most likely observation for each state using the same emission
    # probabiltiies
    seqlen = 100
    count = 1000
    seqs_hidden, seqs, seqs_most_likely = generate_seqs(
        P0,
        P,
        O,
        seqlen,
        count
    )

    # decode each observed sequence
    seqs_decoded, seqs_decoded_most_likely = [], []
    for i, (seq, seq_most_likely) in enumerate(zip(seqs, seqs_most_likely)):
        sys.stdout.write(f"decoding sequences {i+1}\r")
        sys.stdout.flush()

        seqs_decoded.append( predict_state_seq(seq, P0, P, O) )
        seqs_decoded_most_likely.append(
            predict_state_seq(seq_most_likely, P0, P, O)
        )

    print("")

    # find the distance between decoded and ground truth hidden sequences
    dist = lambda seq, ref: np.sum([int(a != b) for a, b in zip(seq, ref)])
    distances = [dist(seq, ref) for seq, ref in zip(seqs_decoded, seqs_hidden)]
    distances_most_likely = [
        dist(seq, ref)
        for seq, ref in zip(seqs_decoded_most_likely, seqs_hidden)
    ]

    print(
        "mean distance between predicted and "
        "correct observed sequences: "
        f"{np.mean(distances):.2f}"
    )
    print(
        "mean distance between predicted and "
        "correct observed most likely sequences: "
        f"{np.mean(distances_most_likely):.2f}"
    )
