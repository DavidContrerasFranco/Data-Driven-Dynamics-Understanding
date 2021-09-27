
import numpy as np
import networkx as nx
import numpy.random as npr
from matplotlib.pyplot import cm
from networkx.utils import py_random_state


def _random_subset(seq, m, rng):
    """
    Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Taken from networkx.generators.random_graphs
    """
    targets = set()
    while len(targets) < m:
        targets.add(rng.choice(seq))
    return list(targets)


def _random_subset_weighted(seq, m, fitness_values, rng):
    """
    Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Modified version of networkx.generators.random_graphs._random_subset
    """
    targets = set()
    weights = np.array([fitness_values[node] for node in seq])
    weights = weights / np.sum(weights)
    while len(targets)<m:
        x = npr.choice(list(seq), 1, p=weights.tolist())
        targets.add(x[0])
    return list(targets)


@py_random_state(2)
def ba_graph_degree(n, m, seed=None):
    """
    Returns a random graph using Barabási–Albert preferential attachment,
    and the degree evolution of the initial nodes with color labels.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Modified version of networkx.generators.random_graphs.barabasi_albert_graph

    Parameters:
    n : int = Number of nodes
    m : int = Number of edges to attach from a new node to existing nodes

    Returns:
    G : Graph
    degree_hist : ndarray = Degree history of the graph
    colors : color labels for plotting
    """
    # M initial nodes not connected to avoid skewing preferential attachment
    # Only the initial number of nodes is relevant (for small t)
    G = nx.empty_graph(m)

    # Helper variables
    repeated_nodes = list(range(m))
    degrees = np.zeros(n, dtype='uint')
    degree_hist = np.zeros((n, n), dtype='int64')
    colors = ['black'] * m + ['deepskyblue']*(n - m)
    

    # Name the graph
    G.name = "BA_Model_({}_{})".format(n,m)

    # Add the other nodes
    for new_node in range(m, n):
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)

        # Add edges to m nodes from the source.
        G.add_edges_from(zip([new_node] * m, targets))

        # Add one node to the list for each new edge just created.
        # & the new node with m edges
        repeated_nodes.extend([new_node] * m + targets)

        # Change degrees values
        degrees[new_node] = m
        degrees[targets] += 1
        degree_hist[new_node] = degrees
    
    return G, degree_hist, colors


@py_random_state(2)
def mixed_graph_degree(n, m, seed=None, keys=None):
    """
    Returns a random graph using randomly the rules of Barabási–Albert
    preferential attachment or simple attachment, and the degree evolution
    of the initial nodes with color labels.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Modified version of networkx.generators.random_graphs.barabasi_albert_graph

    Parameters:
    n : int = Number of nodes
    m : int = Number of edges to attach from a new node to existing nodes

    Returns:
    G : Graph
    degree_hist : ndarray = Degree history of the graph
    colors : color labels for plotting
    """
    # Initialize keys if its None
    if keys == None:
        keys = {
            'A': {
                'attachment': m,
                'rule': 'Preferential Attachment',
                'color': 'deepskyblue'
            },
            'B': {
                'attachment': 1,
                'rule': 'Simple Attachment',
                'color': 'olivedrab'
            }
        }

    # M initial nodes not connected to avoid skewing preferential attachment
    # Only the initial number of nodes is relevant (for small t)
    G = nx.empty_graph(m)

    # Helper variables
    repeated_nodes = list(range(m))
    degrees = np.zeros(n, dtype='uint')
    degree_hist = np.zeros((n, n), dtype='int64')
    colors = ['black'] * m

    # Name the graph
    G.name = "Mixed_Model_({}_{})".format(n,m)

    # Add the other nodes
    for new_node in range(m, n):
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)

        # Add edges to m nodes from the source.
        G.add_edges_from(zip([new_node] * m, targets))

        # Add one node to the list for each new edge just created.
        # & the new node with m edges
        node_type = np.random.choice(list(keys.keys()))
        repeated_nodes.extend([new_node] * keys[node_type]['attachment'] + targets)
        colors.append(keys[node_type]['color'])

        # Change degrees values
        degrees[new_node] = m
        degrees[targets] += 1
        degree_hist[new_node] = degrees
    
    return G, degree_hist, colors


@py_random_state(2)
def ba_fitness_degree(n, m, seed=None):
    """
    Returns a random graph using randomly the rules  of Barabási–Albert
    preferential attachment and from the Erdős-Rényi simple attachment,
    and the degree evolution of the initial nodes with color labels.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Modified version of networkx.generators.random_graphs.barabasi_albert_graph

    Parameters:
    n : int = Number of nodes
    m : int = Number of edges to attach from a new node to existing nodes

    Returns:
    G : Graph
    degree_hist : ndarray = Degree history of the graph
    colors : color labels for plotting
    """
    # M initial nodes not connected to avoid skewing preferential attachment
    # Only the initial number of nodes is relevant (for small t)
    G = nx.empty_graph(m)

    # Helper variables
    fitness_values = {i: value for i, value in enumerate(npr.uniform(0, 1, n))}
    degree_hist = np.zeros((n, n), dtype='int64')
    degrees = np.zeros(n, dtype='uint')
    repeated_nodes = list(range(m))

    # Gradient colors to represent fitness
    fit_vals_lst = list(fitness_values.values())
    colors = [[0., 0., 0.]] * m + cm.Blues(fit_vals_lst)[m:,:3].tolist()
    

    # Name the graph
    G.name = "BA_Fitness_Model_({}_{})".format(n,m)

    # Add the other nodes
    for new_node in range(m, n):
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset_weighted(repeated_nodes, m, fitness_values, seed)

        # Add edges to m nodes from the source.
        G.add_edges_from(zip([new_node] * m, targets))

        # Add one node to the list for each new edge just created.
        # & the new node with m edges
        repeated_nodes.extend([new_node] * m + targets)

        # Change degrees values
        degrees[new_node] = m
        degrees[targets] += 1
        degree_hist[new_node] = degrees
    
    return G, degree_hist, colors


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import matplotlib.patches as mpatches

    def animate(num, G, ax, pos, colors):
        ax.clear()
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(-1.1, 1.1)

        index = range(num)
        H = G.subgraph(index)
        color = colors[:num]
        new_pos = {coor:pos[coor] for coor in index}
        nx.draw(H, pos=new_pos, ax=ax, with_labels=True, node_color=color, font_color="whitesmoke")


    m = 2
    n = 20
    G, degree_hist, colors = mixed_graph_degree(n, m)
    # Relevant array: degree_hist[m:,m]
    
    # Build plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Network base-end values
    n_nodes = G.number_of_nodes()
    pos = nx.circular_layout(G)

    # Make Animation
    ani = animation.FuncAnimation(fig, animate, frames=range(m, n_nodes+1),
                                    interval=500, fargs=(G, ax, pos, colors))

    # Legend for Mixed Network
    preferential = mpatches.Patch(color='deepskyblue', label='Preferential')
    simple = mpatches.Patch(color='olivedrab', label='Simple')
    fig.legend(handles=[preferential, simple])

    # # Legend for Fitness Network
    # fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues))

    # Save Animation as a GIF
    folder_path = os.path.join(os.path.abspath(''), '..', '..', 'Reports', 'Figures')
    filename = os.path.join(folder_path, 'mixed_example.gif')
    ani.save(filename, writer='pillow')

    plt.show()