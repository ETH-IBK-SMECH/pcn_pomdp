import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def load_graph(label, visualize=False):
    graph = pickle.load(open('graphs/graph_{}.pickle'.format(label), 'rb'))

    if visualize:
        points = np.stack([
            np.stack([data['longitude'] for _, data in graph.nodes(data=True)]),
            np.stack([data['latitude'] for _, data in graph.nodes(data=True)]),
        ], axis=1)
        nx.draw(graph, points)
        plt.show()

    return graph
