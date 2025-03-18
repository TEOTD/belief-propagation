import numpy as np

def init_msg_i_to_C(edges, s, t):
    msg_i_to_C = {}

    for (i, j) in edges:
        msg_i_to_C[(i, j)] = {}
        # Message from i to C (i,j)
        if i == s:
            msg_i_to_C[(i, j)][i] = np.array([0.0, 1.0])
        elif i == t:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 0.0])
        else:
            msg_i = np.array([1.0, 1.0])
            msg_i /= msg_i.sum()
            msg_i_to_C[(i, j)][i] = msg_i
        # Message from j to C (i,j)
        if j == s:
            msg_i_to_C[(i, j)][j] = np.array([0.0, 1.0])
        elif j == t:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 0.0])
        else:
            msg_j = np.array([1.0, 1.0])
            msg_j /= msg_j.sum()
            msg_i_to_C[(i, j)][j] = msg_j

    return msg_i_to_C

def max_product(M, w, s, t, its):
    n = M.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if M[i, j] != 0 and i < j]

    # Initialize messages from variables to clusters (edges)
    msg_i_to_C = init_msg_i_to_C(edges, s, t)

    # Initialize messages from clusters to variables
    msg_C_to_i = {}
    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}
        # Message to i
        msg_j = msg_i_to_C[(i, j)][j]
        msg_to_i = np.zeros(2)
        for xi in [0, 1]:
            max_val = -np.inf
            for xj in [0, 1]:
                potential = w[i, j] if xi != xj else 1.0
                val = potential * msg_j[xj]
                if val > max_val:
                    max_val = val
            msg_to_i[xi] = max_val
        sum_msg = msg_to_i.sum()
        msg_C_to_i[(i, j)][i] = msg_to_i / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])
        # Message to j

        msg_i = msg_i_to_C[(i, j)][i]
        msg_to_j = np.zeros(2)
        for xj in [0, 1]:
            max_val = -np.inf
            for xi in [0, 1]:
                potential = w[i, j] if xi != xj else 1.0
                val = potential * msg_i[xi]
                if val > max_val:
                    max_val = val
            msg_to_j[xj] = max_val
        sum_msg = msg_to_j.sum()
        msg_C_to_i[(i, j)][j] = msg_to_j / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])

    for _ in range(its):
        # Compute new cluster to variable messages
        new_msg_C_to_i = {}
        for (i, j) in edges:
            new_msg_C_to_i[(i, j)] = {}
            # Message to i
            msg_j = msg_i_to_C[(i, j)][j]
            msg_to_i = np.zeros(2)
            for xi in [0, 1]:
                max_val = -np.inf
                for xj in [0, 1]:
                    potential = w[i, j] if xi != xj else 1.0
                    val = potential * msg_j[xj]
                    if val > max_val:
                        max_val = val
                msg_to_i[xi] = max_val
            sum_msg = msg_to_i.sum()
            new_msg_C_to_i[(i, j)][i] = msg_to_i / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])
            # Message to j
            msg_i = msg_i_to_C[(i, j)][i]
            msg_to_j = np.zeros(2)
            for xj in [0, 1]:
                max_val = -np.inf
                for xi in [0, 1]:
                    potential = w[i, j] if xi != xj else 1.0
                    val = potential * msg_i[xi]
                    if val > max_val:
                        max_val = val
                msg_to_j[xj] = max_val
            sum_msg = msg_to_j.sum()
            new_msg_C_to_i[(i, j)][j] = msg_to_j / sum_msg if sum_msg != 0 else np.array([0.5, 0.5])

        # Compute new variable to cluster messages
        new_msg_i_to_C = {}
        for (i, j) in edges:
            new_msg_i_to_C[(i, j)] = {}
            # Message from i to C (i,j)
            if i == s:
                new_msg_i_to_C[(i, j)][i] = np.array([0.0, 1.0])
            elif i == t:
                new_msg_i_to_C[(i, j)][i] = np.array([1.0, 0.0])
            else:
                product = np.array([1.0, 1.0])
                connected_edges = [e for e in edges if (e[0] == i or e[1] == i) and e != (i, j)]
                for e in connected_edges:
                    if e[0] == i:
                        product *= new_msg_C_to_i[e][i]
                    else:
                        product *= new_msg_C_to_i[e][i]
                product /= product.sum() if product.sum() != 0 else 1
                new_msg_i_to_C[(i, j)][i] = product
            # Message from j to C (i,j)
            if j == s:
                new_msg_i_to_C[(i, j)][j] = np.array([0.0, 1.0])
            elif j == t:
                new_msg_i_to_C[(i, j)][j] = np.array([1.0, 0.0])
            else:
                product = np.array([1.0, 1.0])
                connected_edges = [e for e in edges if (e[0] == j or e[1] == j) and e != (i, j)]
                for e in connected_edges:
                    if e[0] == j:
                        product *= new_msg_C_to_i[e][j]
                    else:
                        product *= new_msg_C_to_i[e][j]
                product /= product.sum() if product.sum() != 0 else 1
                new_msg_i_to_C[(i, j)][j] = product

        msg_C_to_i = new_msg_C_to_i
        msg_i_to_C = new_msg_i_to_C

    # Compute node beliefs
    node_beliefs = {}
    for i in range(n):
        if i == s:
            node_beliefs[i] = np.array([0.0, 1.0])
        elif i == t:
            node_beliefs[i] = np.array([1.0, 0.0])
        else:
            product = np.array([1.0, 1.0])
            connected_edges = [e for e in edges if e[0] == i or e[1] == i]
            for e in connected_edges:
                if e[0] == i:
                    product *= msg_C_to_i[e][i]
                else:
                    product *= msg_C_to_i[e][i]
            product /= product.sum() if product.sum() != 0 else 1
            node_beliefs[i] = product

    # Determine the assignment
    assignment = []
    for i in range(n):
        if i == s:
            assignment.append(1)
        elif i == t:
            assignment.append(0)
        else:
            belief = node_beliefs[i]
            if np.isclose(belief[0], belief[1], atol=1e-6):
                assignment.append(0.5)
            else:
                assignment.append(0 if belief[0] > belief[1] else 1)

    return assignment
