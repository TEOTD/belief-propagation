import numpy as np


def init_msg_i_to_C(edges, s, t):
    """Initialize node-to-clique messages for max-product.
    - s sends [0,1] (X_s=1), t sends [1,0] (X_t=0).
    - Other nodes start with uniform messages [0.5, 0.5]."""
    msg_i_to_C = {}
    for (i, j) in edges:
        msg_i_to_C[(i, j)] = {}

        # Message from node i to clique (i,j)
        if i == s:
            msg_i_to_C[(i, j)][i] = np.array([0.0, 1.0])  # X_s=1 constraint
        elif i == t:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 0.0])  # X_t=0 constraint
        else:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 1.0]) / 2.0  # Uniform initialization

        # Message from node j to clique (i,j)
        if j == s:
            msg_i_to_C[(i, j)][j] = np.array([0.0, 1.0])  # X_j=1 constraint
        elif j == t:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 0.0])  # X_j=0 constraint
        else:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 1.0]) / 2.0  # Uniform initialization
    return msg_i_to_C


def init_msg_C_to_i(edges, weights, msg_i_to_C):
    """Initialize clique-to-node messages for max-product.
    - For each edge (i,j), compute messages to i and j using max-product rule."""
    msg_C_to_i = {}
    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}

        # Message from clique (i,j) to node i
        msg_j = msg_i_to_C[(i, j)][j]
        msg_to_i = np.zeros(2)
        for xi in [0, 1]:
            max_val = -np.inf
            for xj in [0, 1]:
                # Edge potential: w_ij if xi ≠ xj, else 1.0
                potential = weights[i, j] if xi != xj else 1.0
                val = potential * msg_j[xj]
                max_val = max(max_val, val)
            msg_to_i[xi] = max_val
        sum_msg = msg_to_i.sum()
        msg_C_to_i[(i, j)][i] = msg_to_i / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])

        # Message from clique (i,j) to node j
        msg_i = msg_i_to_C[(i, j)][i]
        msg_to_j = np.zeros(2)
        for xj in [0, 1]:
            max_val = -np.inf
            for xi in [0, 1]:
                potential = weights[i, j] if xi != xj else 1.0
                val = potential * msg_i[xi]
                max_val = max(max_val, val)
            msg_to_j[xj] = max_val
        sum_msg = msg_to_j.sum()
        msg_C_to_i[(i, j)][j] = msg_to_j / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])
    return msg_C_to_i


def updt_msg_C_to_i(edges, weights, msg_i_to_C):
    """Update clique-to-node messages iteratively using max-product."""
    msg_C_to_i = {}
    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}

        # Message to node i
        msg_j = msg_i_to_C[(i, j)][j]
        msg_to_i = np.zeros(2)
        for xi in [0, 1]:
            max_val = -np.inf
            for xj in [0, 1]:
                potential = weights[i, j] if xi != xj else 1.0
                val = potential * msg_j[xj]
                max_val = max(max_val, val)
            msg_to_i[xi] = max_val
        sum_msg = msg_to_i.sum()
        msg_C_to_i[(i, j)][i] = msg_to_i / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])

        # Message to node j
        msg_i = msg_i_to_C[(i, j)][i]
        msg_to_j = np.zeros(2)
        for xj in [0, 1]:
            max_val = -np.inf
            for xi in [0, 1]:
                potential = weights[i, j] if xi != xj else 1.0
                val = potential * msg_i[xi]
                max_val = max(max_val, val)
            msg_to_j[xj] = max_val
        sum_msg = msg_to_j.sum()
        msg_C_to_i[(i, j)][j] = msg_to_j / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])
    return msg_C_to_i


def updt_msg_i_to_C(edges, s, t, msg_C_to_i):
    """Update node-to-clique messages by multiplying incoming clique-to-node messages."""
    msg_i_to_C = {}
    for (i, j) in edges:
        msg_i_to_C[(i, j)] = {}

        # Message from node i to clique (i,j)
        if i == s:
            msg_i_to_C[(i, j)][i] = np.array([0.0, 1.0])  # X_i=1 constraint
        elif i == t:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 0.0])  # X_i=0 constraint
        else:
            product = np.array([1.0, 1.0])
            # Multiply messages from all connected cliques except (i,j)
            connected_edges = [e for e in edges if (e[0] == i or e[1] == i) and e != (i, j)]
            for e in connected_edges:
                product *= msg_C_to_i[e][i]
            sum_prod = product.sum()
            msg_i_to_C[(i, j)][i] = product / sum_prod if sum_prod != 0 else np.array([0.5, 0.5])

        # Message from node j to clique (i,j)
        if j == s:
            msg_i_to_C[(i, j)][j] = np.array([0.0, 1.0])  # X_j=1 constraint
        elif j == t:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 0.0])  # X_j=0 constraint
        else:
            product = np.array([1.0, 1.0])
            connected_edges = [e for e in edges if (e[0] == j or e[1] == j) and e != (i, j)]
            for e in connected_edges:
                product *= msg_C_to_i[e][j]
            sum_prod = product.sum()
            msg_i_to_C[(i, j)][j] = product / sum_prod if sum_prod != 0 else np.array([0.5, 0.5])
    return msg_i_to_C


def calculate_node_beliefs(n, edges, s, t, msg_C_to_i):
    """Compute node beliefs by multiplying all incoming clique-to-node messages."""
    node_beliefs = {}
    for i in range(n):
        if i == s:
            node_beliefs[i] = np.array([0.0, 1.0])  # X_s=1
        elif i == t:
            node_beliefs[i] = np.array([1.0, 0.0])  # X_t=0
        else:
            product = np.array([1.0, 1.0])
            connected_edges = [e for e in edges if e[0] == i or e[1] == i]
            for e in connected_edges:
                product *= msg_C_to_i[e][i]  # Multiply all incoming messages
            sum_prod = product.sum()
            node_beliefs[i] = product / sum_prod if sum_prod != 0 else np.array([0.5, 0.5])
    return node_beliefs


def calc_assignment(node_beliefs, n, s, t):
    """Convert node beliefs to 0/1 assignments (0.5 for ties)."""
    assignment = []
    for i in range(n):
        if i == s:
            assignment.append(1)  # X_s=1
        elif i == t:
            assignment.append(0)  # X_t=0
        else:
            belief = node_beliefs[i]
            # Check for near-tie (tolerance 1e-6)
            if np.isclose(belief[0], belief[1], atol=1e-6):
                assignment.append(0.5)
            else:
                assignment.append(0 if belief[0] > belief[1] else 1)
    return assignment


def max_product(adjacency_matrix, weights, s, t, its):
    """Max-product algorithm for s-t cut.
    Steps:
    1. Extract edges from adjacency matrix.
    2. Initialize messages.
    3. Iteratively update messages for `its` iterations.
    4. Compute node beliefs and return assignments."""
    n = adjacency_matrix.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if adjacency_matrix[i, j] != 0 and i < j]

    # Initialize messages
    msg_i_to_C = init_msg_i_to_C(edges, s, t)
    msg_C_to_i = init_msg_C_to_i(edges, weights, msg_i_to_C)

    # Iterate message updates
    for _ in range(its):
        msg_C_to_i = updt_msg_C_to_i(edges, weights, msg_i_to_C)
        msg_i_to_C = updt_msg_i_to_C(edges, s, t, msg_C_to_i)

    # Compute final assignments
    node_beliefs = calculate_node_beliefs(n, edges, s, t, msg_C_to_i)
    return calc_assignment(node_beliefs, n, s, t)