"""Omplementation of the expectation maximization algorithm for learning in
HMMs.
"""


import numpy as np


from hmm import alpha, beta, xi, gamma


class EM():
    def __init__(
            self,
            nof_hidden_states: int,
            nof_observed_states: int,
            seqs: list
    ):
        """Initializes the model.

        Args:
            nof_hidden_states:   number of hidden states
            nof_observed_states: number of observed states
            seqs:                list of observed sequences to train on
        """

        self.seqs = seqs

        # Initial hidden state probabilities
        self.P0 = np.random.rand(nof_hidden_states)

        # Hidden state transition probabilities
        self.P = np.random.rand(nof_hidden_states, nof_hidden_states)

        # Emission probabilities
        self.O = np.random.rand(nof_hidden_states, nof_observed_states)


    def iterate(self):
        """Performs a single expectation and a single maximization step.
        """

        # Expectation step - initialize sums
        P0_sum_num = np.zeros(self.P0.shape)
        P_sum_num, P_sum_den = np.zeros(self.P.shape), np.zeros(self.P.shape[0])
        O_sum_num, O_sum_den = np.zeros(self.O.shape), np.zeros(self.O.shape[0])

        # Expectation step - aggregate contribution of each sequence
        for seq in self.seqs:
            # Fwd/backward functions
            alpha_matrix = alpha(seq, self.P0, self.P, self.O)
            beta_matrix = beta(seq, self.P0, self.P, self.O)

            # Baum Welch functions
            xi_matrix = xi(seq, alpha_matrix, beta_matrix, self.P, self.O)
            gamma_matrix = gamma(alpha_matrix, beta_matrix)

            # Update sums
            P0_sum_num = P0_sum_num + gamma_matrix[0]

            P_sum_num = P_sum_num + np.sum(xi_matrix, axis=0)
            P_sum_den = P_sum_den + np.sum(gamma_matrix[:-1], axis=0)
   
            for t in range(gamma_matrix.shape[0]):
                O_sum_num[:,seq[t]] = O_sum_num[:,seq[t]] + gamma_matrix[t]
            O_sum_den = O_sum_den + np.sum(gamma_matrix, axis=0).reshape(-1, 1)

        # Maximization step
        self.P0 = P0_sum_num / len(self.seqs)
        self.P = P_sum_num / P_sum_den.reshape(-1, 1)
        self.O = O_sum_num / O_sum_den
