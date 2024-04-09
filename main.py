import numpy as np
from pcn_graphs import load_graph
from data_simulation import mdp_sequences


if __name__ == '__main__':

    # Load PCN sub-loop
    loop_graph = load_graph(38, visualize=False)
    print(loop_graph)

    # degradation parameters
    n_nodes = len(loop_graph.nodes)
    n_states = 3
    p_degradation = 5e-2

    # degradation transition
    p_transition_1 = np.diag(np.ones(n_states) - p_degradation) + np.diag(np.ones(n_states - 1) * p_degradation, 1)
    p_transition_1[-1, -1] = 1.  # last state cannot transition to anything else

    # action transition
    p_transition_2 = np.tril(np.tile(np.arange(n_states, 0, -1), (n_states, 1))) ** 2
    p_transition_2 = (p_transition_2.T / p_transition_2.sum(1)).T

    p_transitions = np.array([p_transition_1, p_transition_2])

    print("Transition matrix of HMM process:")
    print(p_transitions)

    # homoscedastic emission process
    mus = np.arange(n_states)
    # uncertainty is the highest when the state is half way between the two extremes
    min_var = 0.1
    max_var = 1.
    sigmas = (min_var + (max_var - min_var) * np.sin(np.pi * np.arange(n_states) / (n_states-1))**2)

    # generate data points
    sequences = mdp_sequences(loop_graph, p_transitions, mus, sigmas, num_iterations=1000)

    print(sequences)

    # define model

    # MCMC inference

    # plot results
