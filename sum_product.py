import numpy as np


def init_msg_i_to_C(edges, s, t):
    """Initialize node-to-clique messages.
    - For edges (i,j), set messages from nodes i/j to clique (i,j).
    - s and t are constrained: s always sends [0,1], t sends [1,0].
    - Other nodes start with uniform messages [0.5, 0.5]."""
    msg_i_to_C = {}
    for (i, j) in edges:
        msg_i_to_C[(i, j)] = {}

        # Message from node i to clique (i,j)
        if i == s:
            msg_i_to_C[(i, j)][i] = np.array([0.0, 1.0])  # X_i = s = 1
        elif i == t:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 0.0])  # X_i = t = 0
        else:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 1.0]) / 2.0  # Uniform initialization

        # Message from node j to clique (i,j)
        if j == s:
            msg_i_to_C[(i, j)][j] = np.array([0.0, 1.0])  # X_j = s = 1
        elif j == t:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 0.0])  # X_j = t = 0
        else:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 1.0]) / 2.0  # Uniform initialization

    return msg_i_to_C


def init_msg_C_to_i(edges, weights, msg_i_to_C):
    """Initialize clique-to-node messages.
    - For edge (i,j), compute messages from clique (i,j) to node i/j.
    - Use edge potential psi(Xi, Xj) = w_ij if Xi ≠ Xj, else 1.0.
    - Messages are normalized to sum to 1."""
    msg_C_to_i = {}
    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}

        # Message from clique (i,j) to node i
        msg_j = msg_i_to_C[(i, j)][j]  # Message from node j to clique (i,j)
        msg_Xi_0 = 1.0 * msg_j[0] + weights[i, j] * msg_j[1]  # Xi=0: sum over Xj with psi(0,Xj)
        msg_Xi_1 = weights[i, j] * msg_j[0] + 1.0 * msg_j[1]  # Xi=1: sum over Xj with psi(1,Xj)
        sum_Xi = msg_Xi_0 + msg_Xi_1
        msg_C_to_i[(i, j)][i] = np.array([msg_Xi_0, msg_Xi_1]) / (sum_Xi + 1e-12)  # Normalize

        # Message from clique (i,j) to node j
        msg_i = msg_i_to_C[(i, j)][i]  # Message from node i to clique (i,j)
        msg_Xj_0 = 1.0 * msg_i[0] + weights[i, j] * msg_i[1]  # Xj=0: sum over Xi with psi(Xi,0)
        msg_Xj_1 = weights[i, j] * msg_i[0] + 1.0 * msg_i[1]  # Xj=1: sum over Xi with psi(Xi,1)
        sum_Xj = msg_Xj_0 + msg_Xj_1
        msg_C_to_i[(i, j)][j] = np.array([msg_Xj_0, msg_Xj_1]) / (sum_Xj + 1e-12)  # Normalize

    return msg_C_to_i

def updt_msg_C_to_i(edges, weights, s, t, msg_i_to_C):
    """Update clique-to-node messages iteratively.
    - Recompute messages using latest node-to-clique messages.
    - Enforce constraints for s (always [0,1]) and t (always [1,0])."""
    msg_C_to_i = {}
    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}

        # Message to node i
        if i == s:
            msg_C_to_i[(i, j)][i] = np.array([0.0, 1.0])  # Force X_s = 1
        elif i == t:
            msg_C_to_i[(i, j)][i] = np.array([1.0, 0.0])  # Force X_t = 0
        else:
            # Compute message from clique (i,j) to node i using node j's messages
            msg_j = msg_i_to_C[(i, j)].get(j, np.array([1.0, 1.0]))
            msg_Xi_0 = 1.0 * msg_j[0] + weights[i, j] * msg_j[1]
            msg_Xi_1 = weights[i, j] * msg_j[0] + 1.0 * msg_j[1]
            sum_Xi = msg_Xi_0 + msg_Xi_1
            msg_C_to_i[(i, j)][i] = np.array([msg_Xi_0, msg_Xi_1]) / (sum_Xi + 1e-12)

        # Message to node j
        if j == s:
            msg_C_to_i[(i, j)][j] = np.array([0.0, 1.0])  # Force X_j = 1
        elif j == t:
            msg_C_to_i[(i, j)][j] = np.array([1.0, 0.0])  # Force X_j = 0
        else:
            # Compute message from clique (i,j) to node j using node i's messages
            msg_i = msg_i_to_C[(i, j)].get(i, np.array([1.0, 1.0]))
            msg_Xj_0 = 1.0 * msg_i[0] + weights[i, j] * msg_i[1]
            msg_Xj_1 = weights[i, j] * msg_i[0] + 1.0 * msg_i[1]
            sum_Xj = msg_Xj_0 + msg_Xj_1
            msg_C_to_i[(i, j)][j] = np.array([msg_Xj_0, msg_Xj_1]) / (sum_Xj + 1e-12)

    return msg_C_to_i


def updt_msg_i_to_C(edges, s, t, msg_C_to_i):
    """Update node-to-clique messages iteratively.
    - Messages are products of all incoming clique-to-node messages except from the target clique."""
    msg_i_to_C = {}
    for (i, j) in edges:
        msg_i_to_C[(i, j)] = {}

        # Message from node i to clique (i,j)
        edges_connected_to_i = [e for e in edges if (e[0] == i or e[1] == i) and e != (i, j)]
        product = np.array([1.0, 1.0])
        for edge in edges_connected_to_i:
            product *= msg_C_to_i[edge].get(i, np.array([0.5, 0.5]))  # Multiply messages from other cliques
        sum_prod = product.sum() + 1e-12
        if i == s:
            msg_i_to_C[(i, j)][i] = np.array([0.0, 1.0])  # Override for s
        elif i == t:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 0.0])  # Override for t
        else:
            msg_i_to_C[(i, j)][i] = product / sum_prod  # Normalize

        # Message from node j to clique (i,j)
        edges_connected_to_j = [e for e in edges if (e[0] == j or e[1] == j) and e != (i, j)]
        product = np.array([1.0, 1.0])
        for edge in edges_connected_to_j:
            product *= msg_C_to_i[edge].get(j, np.array([0.5, 0.5]))  # Multiply messages from other cliques
        sum_prod = product.sum() + 1e-12
        if j == s:
            msg_i_to_C[(i, j)][j] = np.array([0.0, 1.0])  # Override for s
        elif j == t:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 0.0])  # Override for t
        else:
            msg_i_to_C[(i, j)][j] = product / sum_prod  # Normalize

    return msg_i_to_C


def calculate_node_beliefs(n, edges, s, t, msg_C_to_i):
    """Compute marginal beliefs for each node.
    - Beliefs are products of all incoming clique-to-node messages.
    - Hard constraints enforced for s and t."""
    nodes_beliefs = {}
    for i in range(n):
        connected_edges = [e for e in edges if e[0] == i or e[1] == i]
        product = np.array([1.0, 1.0])
        for edge in connected_edges:
            product *= msg_C_to_i[edge].get(i, np.array([0.5, 0.5]))  # Aggregate messages
        sum_product = product.sum() + 1e-12
        if i == s:
            nodes_beliefs[i] = np.array([0.0, 1.0])  # X_s = 1
        elif i == t:
            nodes_beliefs[i] = np.array([1.0, 0.0])  # X_t = 0
        else:
            nodes_beliefs[i] = product / sum_product  # Normalize for other nodes
    return nodes_beliefs


def calculate_edge_beliefs(edges, weights, msg_i_to_C):
    """Compute joint beliefs for edges.
    - Beliefs combine edge potentials and node-to-clique messages."""
    edge_beliefs = {}
    for (i, j) in edges:
        belief = np.zeros((2, 2))
        for Xi in [0, 1]:
            for Xj in [0, 1]:
                # Edge potential: w_ij if Xi ≠ Xj, else 1.0
                edge_potential = weights[i, j] if Xi != Xj else 1.0
                # Multiply messages from nodes i and j
                belief[Xi, Xj] = edge_potential * msg_i_to_C[(i, j)][i][Xi] * msg_i_to_C[(i, j)][j][Xj]
        sum_belief = belief.sum() + 1e-12
        edge_beliefs[(i, j)] = belief / sum_belief  # Normalize
    return edge_beliefs


def calculate_beth_free_energy(nodes_beliefs, edge_beliefs, weights, edges):
    """Compute Bethe free energy approximation of the partition function Z.
    - Free energy = node_entropies - edge_energies + mutual_information."""
    free_energy = 0.0

    # Node entropies: sum_i [belief_i * log(belief_i)]
    for i in nodes_beliefs:
        belief = nodes_beliefs[i] + 1e-12  # Avoid log(0)
        free_energy += np.sum(belief * np.log(belief))

    # Edge energies and mutual information
    for (i, j) in edges:
        belief_ij = edge_beliefs[(i, j)] + 1e-12
        for Xi in [0, 1]:
            for Xj in [0, 1]:
                edge_potential = weights[i, j] if Xi != Xj else 1.0
                # Subtract edge energy: belief_ij * log(psi)
                free_energy -= belief_ij[Xi, Xj] * np.log(edge_potential + 1e-12)
                # Add mutual information: belief_ij * log(belief_ij / (belief_i * belief_j))
                belief_i = nodes_beliefs[i][Xi] + 1e-12
                belief_j = nodes_beliefs[j][Xj] + 1e-12
                mutual_info = belief_ij[Xi, Xj] * np.log(belief_ij[Xi, Xj] / (belief_i * belief_j))
                free_energy += mutual_info

    return free_energy


def sum_product(A, s, t, w, its):
    """Main function to compute the approximate partition function using sum-product.
    Steps:
    1. Extract edges from the adjacency matrix.
    2. Initialize messages.
    3. Iteratively update messages for `its` iterations.
    4. Compute node and edge beliefs.
    5. Calculate Bethe free energy and approximate Z."""
    n = A.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if A[i, j] != 0 and i < j]

    # Initialize messages
    msg_i_to_C = init_msg_i_to_C(edges, s, t)
    msg_C_to_i = init_msg_C_to_i(edges, w, msg_i_to_C)

    # Iterate message updates
    for _ in range(its):
        msg_C_to_i = updt_msg_C_to_i(edges, w, s, t, msg_i_to_C)
        msg_i_to_C = updt_msg_i_to_C(edges, s, t, msg_C_to_i)

    # Compute beliefs and free energy
    nodes_beliefs = calculate_node_beliefs(n, edges, s, t, msg_C_to_i)
    edge_beliefs = calculate_edge_beliefs(edges, w, msg_i_to_C)
    free_energy = calculate_beth_free_energy(nodes_beliefs, edge_beliefs, w, edges)
    Z = np.exp(-free_energy)

    return round(Z, 0), edge_beliefs  # Return integer approximation of Z