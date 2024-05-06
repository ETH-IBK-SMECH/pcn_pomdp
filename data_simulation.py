import random
import numpy as np
import networkx as nx
from scipy.stats import multinomial, norm
from tqdm import tqdm


class Agent:
    """
    This class handles the roaming of the agent along the graph.
    """
    def __init__(self, graph):
        self.graph = graph
        self.end_nodes = [x for x in self.graph.nodes() if self.graph.degree(x) == 1]
        self.start_node = random.sample(self.end_nodes, 1)[0]
        self.current_idx = 0

        # for last_node to be different
        self.last_node = self.start_node
        while self.last_node == self.start_node:
            self.last_node = random.sample(self.end_nodes, 1)[0]

        # calculate path
        self.loop_path = nx.astar_path(self.graph, self.start_node, self.last_node)

    # once we each the end of the path, find a new path
    def recalculate_path(self):
        self.start_node = self.last_node
        while self.last_node == self.start_node:
            self.last_node = random.sample(self.end_nodes, 1)[0]
        self.loop_path = nx.astar_path(self.graph, self.start_node, self.last_node)
        self.current_idx = 0


def markov_update(p_trans, current_state):
    p_tr = p_trans[current_state]
    new_state = list(multinomial.rvs(1, p_tr)).index(1)
    return new_state


def mdp_sequences(loop_graph, p_transition, mus, sigmas, num_iterations):

    n_nodes = len(loop_graph.nodes)
    n_states = p_transition.shape[1]

    n_agents = 3

    agents = [Agent(loop_graph) for _ in range(n_agents)]

    # Start simulation
    all_emissions = np.full((n_nodes, num_iterations), np.nan)
    all_states = np.full((n_nodes, num_iterations), 0, dtype=int)
    all_actions = np.full((n_nodes, num_iterations), 0, dtype=int)  # no action equals worse state

    # initial states
    states = np.zeros(n_nodes).astype(int)

    for iteration in tqdm(np.arange(num_iterations)):
        # transition update
        for i in range(n_nodes):
            states[i] = markov_update(p_transition[0], states[i])

        for agent in agents:
            # location information
            current_state = states[agent.loop_path[agent.current_idx]]

            # emission (Homoscedastic gaussian)
            emission = norm.rvs(loc=mus[current_state], scale=sigmas[current_state])
            emission = np.clip(int(emission), 0, n_states - 1)

            # update chain
            all_emissions[agent.loop_path[agent.current_idx], iteration] = emission

            if emission == (n_states - 1):  # if the worst state is detected, repair
                action = 1
                all_actions[agent.loop_path[agent.current_idx], iteration] = action
                states[agent.loop_path[agent.current_idx]] = markov_update(p_transition[action], states[agent.loop_path[agent.current_idx]])

            # get new position info
            agent.current_idx += 1
            if agent.loop_path[agent.current_idx] == agent.last_node:
                agent.recalculate_path()

        all_states[:, iteration] = states

    return {
        'states': all_states,
        'actions': all_actions,
        'emissions': all_emissions
    }
