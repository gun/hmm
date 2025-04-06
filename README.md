The modules here capture an exploratory effort on Hidden Markov Models. The
modules are not organized in any complex hierarchical package structure.

There are 3 types of modules here:

1. Algorithmic functionality:

   `hmm.py`    - probablistic HMM functions

   `em.py`     - the expectation maximization algorithm for learning HMM
                 parameters

   `viterbi.py`- the Viterbi algorithm for finding the most likely hidden
                 sequence


2. Model and data:

   `model.py`  - specification of a sample HMM for testing

   `data.py`   - functions for generating sample sequences from the model


3. Test scripts:

   `decode.py` - runs the Viterbi algorithm to find most likely hidden
                 sequences for a data set of observed sequences.
                 run with `python decode.py`

   `learn.py`  - runs the Baum-Welch algorithm to learn the parameters of a
                 HMM given a data set of observed sequences. run with
                 `python learn.py`
