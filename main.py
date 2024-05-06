import numpy as np
import pymc3 as pm
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()
import arviz as az
import matplotlib.pyplot as plt
from pcn_graphs import load_graph
from data_simulation import mdp_sequences
from models import define_model


if __name__ == '__main__':

    # Use the code below to inspect prior transition matrices
    """
    # degradation prior
    n_states = 3
    min_prior = 0.2
    a = np.full(shape=(n_states, n_states), fill_value=min_prior) + np.diag((7 - min_prior) * np.ones(n_states)) + np.diag(
        (1 - min_prior) * np.ones(n_states - 1), k=1),
    samples_0 = pm.Dirichlet.dist(a=a).random(size=5000)
    az.plot_posterior(samples_0[np.newaxis])
    plt.show()

    # repair prior
    a = np.full(shape=(n_states, n_states), fill_value=min_prior) + np.tril(
        np.arange(n_states ** 2)[::-1].reshape(n_states, n_states).T + 1 - min_prior),
    samples_0 = pm.Dirichlet.dist(a=a).random(size=5000)
    az.plot_posterior(samples_0[np.newaxis])
    plt.show()
    exit()
    #"""

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
    min_var = 1e-3
    max_var = 1e-2
    sigmas = (min_var + (max_var - min_var) * np.sin(np.pi * np.arange(n_states) / (n_states-1))**2)

    # generate data points
    sequences = mdp_sequences(loop_graph, p_transitions, mus, sigmas, num_iterations=1000)

    # define model
    model = define_model(n_states, 2, sequences)

    # MCMC inference
    with model:
        nuts_step = pm.NUTS([model.sigma, model.mu, model.p_transition, model.init_probs], target_accept=0.9)
        trace = pm.sample(2000, step=[nuts_step], return_inferencedata=False)

    # plot results
    az.plot_trace(trace, var_names=["p_transition"], divergences=False)
    plt.show()

    az.plot_trace(trace, var_names=["mu", "sigma"], divergences=False)
    plt.show()

    az.plot_forest(trace, var_names=["mu", "sigma"])
    plt.show()

    az.plot_posterior(trace, var_names=["p_transition"])
    plt.show()

    select_idx = 0
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.plot(np.round(trace["hmm_states_{}".format(select_idx)].mean(axis=0)), label="estimates")
    plt.plot(np.array(sequences['states'][select_idx]), label="true")
    plt.legend()
    plt.show()
