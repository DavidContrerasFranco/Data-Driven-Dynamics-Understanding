
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
        x = rng.choice(seq)
        targets.add(x)
    return list(targets)


def _random_subset_weighted(seq, m, fitness_values, rng):
    """
    Return m unique elements from seq weighted by fitness_values.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Modified version of networkx.generators.random_graphs._random_subset
    """
    targets = set()
    weights = [fitness_values[node] for node in seq]
    while len(targets) < m:
        x = rng.choices(list(seq), weights=weights, k=m-len(targets))
        targets.update(x)
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
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m + 1 <= n, m = {m}, n = {n}"
        )
    # M +1 initial nodes fully connected to avoid skewing preferential attachment
    # Only the initial number of nodes is relevant (for small t)
    G = nx.complete_graph(m+1)

    # Pool of attachment options
    repeated_nodes = list(range(m+1))*m

    # Degree initialization
    degree_hist = np.zeros((n, n), dtype=np.uint32)
    degrees = np.zeros(n, dtype=np.uint32)
    degrees[:m+1] = m
    degree_hist[m] = degrees

    # Color labels for plotting
    colors = ['deepskyblue']*n

    # Name the graph
    G.name = "BA_Model_({}_{})".format(n,m)

    # Add the other nodes
    for new_node in range(m+1, n):
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
    
    return G, degree_hist[m:], colors


@py_random_state(2)
def mixed_graph_degree(n, m, seed=None, init_node=None, keys=None):
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

    if init_node == None:
        for init_node in keys: break

    # M +1 initial nodes fully connected to avoid skewing preferential attachment
    # Only the initial number of nodes is relevant (for small t)
    G = nx.complete_graph(m+1)

    # Pool of attachment options & node_types
    repeated_nodes = list(range(m+1))*keys[init_node]['attachment']
    node_types = [init_node]*(m+1)

    # Degree initialization
    degree_hist = np.zeros((n, n), dtype=np.uint32)
    degrees = np.zeros(n, dtype=np.uint32)
    degrees[:m+1] = m
    degree_hist[m] = degrees

    # Color labels for plotting
    colors = [keys[init_node]['color']] * (m+1)

    # Name the graph
    G.name = "Mixed_Model_({}_{})".format(n,m)

    # Add the other nodes
    for new_node in range(m+1, n):
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)

        # Add edges to m nodes from the source.
        G.add_edges_from(zip([new_node] * m, targets))

        # Add one node to the list for each new edge just created.
        # & the new node with m edges
        node_type = np.random.choice(list(keys.keys()))
        colors.append(keys[node_type]['color'])
        node_types.extend(node_type)

        # print(targets, np.array(node_types)[targets], np.array(targets)[np.array(node_types)[targets] == 'A'])
        expand_targets = np.array(targets)[np.array(node_types)[targets] == 'A'].tolist()
        repeated_nodes.extend([new_node] * keys[node_type]['attachment'] + expand_targets)

        # Change degrees values
        degrees[new_node] = m
        degrees[targets] += 1
        degree_hist[new_node] = degrees

    # # Test to be certain about mixed pool
    # from pprint import pprint
    # from collections import Counter
    # counts = dict(Counter(repeated_nodes))
    # pprint(counts, width=10)

    # print((np.array(list(counts.values()))[np.array(range(n))[np.array(node_types) == 'B']] > 1).any())

    return G, degree_hist[m:], colors


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
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert w/ fitness network must have m >= 1 and m + 1 <= n, m = {m}, n = {n}"
        )
    # M +1 initial nodes fully connected to avoid skewing preferential attachment
    # Only the initial number of nodes is relevant (for small t)
    G = nx.complete_graph(m+1)

    # Pool of attachment options & node_types
    repeated_nodes = list(range(m+1))*m

    # Degree initialization
    degree_hist = np.zeros((n, n), dtype=np.uint32)
    degrees = np.zeros(n, dtype=np.uint32)
    degrees[:m+1] = m
    degree_hist[m] = degrees

    # Fitness
    fitness_values = {i: value for i, value in enumerate(npr.uniform(0, 1, n))}

    # Gradient colors to represent fitness
    fit_vals_lst = list(fitness_values.values())
    colors = cm.viridis(fit_vals_lst)[:,:3].tolist()
    

    # Name the graph
    G.name = "BA_Fitness_Model_({}_{})".format(n,m)

    # Add the other nodes
    for new_node in range(m+1, n):
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
    
    return G, degree_hist[m:], colors


@py_random_state(2)
def ba_discrete_fitness_degree(n, m, seed=None, fitness_levels=[0.991, 0.223],
                                                fitness_values=None):
    """
    Returns a random graph using randomly the rule of Barabási–Albert
    preferential attachment with fitness, and the degree evolution of the
    initial nodes with color labels.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree
    and with higher fitness.

    Modified version of networkx.generators.random_graphs.barabasi_albert_graph

    Parameters:
    n : int = Number of nodes
    m : int = Number of edges to attach from a new node to existing nodes
    fitness_levels : [float] = Discrete levels of fitness to be assigned randomly
    fitness_levels : {int:float} = Assigned values of fitness

    Returns:
    G : Graph
    degree_hist : ndarray = Degree history of the graph
    colors : color labels for plotting
    """
    # M +1 initial nodes fully connected to avoid skewing preferential attachment
    # Only the initial number of nodes is relevant (for small t)
    G = nx.complete_graph(m+1)

    # Pool of attachment options & node_types
    repeated_nodes = list(range(m+1))*m

    # Degree initialization
    degree_hist = np.zeros((n, n), dtype=np.uint32)
    degrees = np.zeros(n, dtype=np.uint32)
    degrees[:m+1] = m
    degree_hist[m] = degrees

    # Fitness
    if fitness_values is None:
        fitness_values = {i: value for i, value in enumerate(npr.choice(fitness_levels, n))}
        fitness_values[0] = fitness_levels[0]

    # Gradient colors to represent fitness
    fit_vals_lst = list(fitness_values.values())
    colors = cm.Blues(fit_vals_lst)[:,:3].tolist()
    

    # Name the graph
    G.name = "BA_Fitness_Model_({}_{})".format(n,m)

    # Add the other nodes
    for new_node in range(m+1, n):
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
    
    return G, degree_hist[m:], colors


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
    G, degree_hist, colors = ba_graph_degree(n, m)#, start_fitness=0.991)#, init_node='B')
    # Relevant array: degree_hist[:,0]
    print(degree_hist)
    
    # Build plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Network base-end values
    pos = nx.circular_layout(G)

    # Make Animation
    ani = animation.FuncAnimation(fig, animate, frames=range(m+1, n+1),
                                    interval=500, fargs=(G, ax, pos, colors))

    # # Legend for Mixed Network
    # preferential = mpatches.Patch(color='deepskyblue', label='Preferential')
    # simple = mpatches.Patch(color='olivedrab', label='Simple')
    # fig.legend(handles=[preferential, simple])

    # # Legend for Fitness Network
    # fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues))

    # Save Animation as a GIF
    folder_path = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'Reports', 'Figures')
    filename = os.path.abspath(os.path.join(folder_path, 'alt_generator.gif'))
    ani.save(filename, writer='pillow')

    plt.show()