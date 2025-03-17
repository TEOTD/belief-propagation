import numpy as np

def init_msg_i_to_C(edges):
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


def init_msg_C_to_i(edges, msg_i_to_C):
    msg_C_to_i = {}

    # Initialize messages from clusters to variables
    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}
        # Message to i
        msg_j = msg_i_to_C[(i, j)][j]
        msg_Xi_0 = 1.0 * msg_j[0] + w[i, j] * msg_j[1]
        msg_Xi_1 = w[i, j] * msg_j[0] + 1.0 * msg_j[1]
        sum_Xi = msg_Xi_0 + msg_Xi_1
        if sum_Xi == 0:
            msg_C_to_i[(i, j)][i] = np.array([0.5, 0.5])
        else:
            msg_C_to_i[(i, j)][i] = np.array([msg_Xi_0, msg_Xi_1]) / sum_Xi
        # Message to j
        msg_i = msg_i_to_C[(i, j)][i]
        msg_Xj_0 = 1.0 * msg_i[0] + w[i, j] * msg_i[1]
        msg_Xj_1 = w[i, j] * msg_i[0] + 1.0 * msg_i[1]
        sum_Xj = msg_Xj_0 + msg_Xj_1
        if sum_Xj == 0:
            msg_C_to_i[(i, j)][j] = np.array([0.5, 0.5])
        else:
            msg_C_to_i[(i, j)][j] = np.array([msg_Xj_0, msg_Xj_1]) / sum_Xj

    return msg_C_to_i

def calculate_node_beliefs(n, edges, msg_C_to_i):
    nodes_beliefs = {}
    for i in range(n):
        connected_edges = [e for e in edges if e[0] == i or e[1] == i]
        product = np.array([1.0, 1.0])
        for edge in connected_edges:
            if edge[0] == i:
                msg = msg_C_to_i[edge].get(i, np.array([0.5, 0.5]))
            else:
                msg = msg_C_to_i[edge].get(i, np.array([0.5, 0.5]))
            product *= msg
        if i == s:
            product = np.array([0.0, product[1]])
        elif i == t:
            product = np.array([product[0], 0.0])
        sum_of_product = product.sum()
        if sum_of_product == 0:
            nodes_beliefs[i] = np.array([0.5, 0.5])
        else:
            nodes_beliefs[i] = product / sum_of_product
    return nodes_beliefs

def calculate_edge_beliefs(edges, msg_i_to_C):
    edge_beliefs = {}
    for (i, j) in edges:
        belief = np.zeros((2, 2))
        for Xi in [0, 1]:
            for Xj in [0, 1]:
                edge_potential = w[i, j] if Xi != Xj else 1.0
                msg_i = msg_i_to_C[(i, j)].get(i, np.array([0.5, 0.5]))[Xi]
                msg_j = msg_i_to_C[(i, j)].get(j, np.array([0.5, 0.5]))[Xj]
                belief[Xi, Xj] = edge_potential * msg_i * msg_j
        sum_belief = belief.sum()
        if sum_belief == 0:
            edge_beliefs[(i, j)] = np.ones((2, 2)) / 4
        else:
            edge_beliefs[(i, j)] = belief / sum_belief
    return edge_beliefs

def sum_product(M, w, s, t, its):
    n = M.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if M[i, j] != 0 and i < j]

    # Initialize messages from variables to clusters (edges)
    msg_i_to_C = init_msg_i_to_C(edges)
    msg_C_to_i = init_msg_C_to_i(edges, msg_i_to_C)

    for _ in range(its):
        # Compute new cluster to variable messages based on previous variable to cluster messages
        new_msg_C_to_i = {}
        for (i, j) in edges:
            new_msg_C_to_i[(i, j)] = {}
            # Message to i
            msg_j = msg_i_to_C[(i, j)].get(j, np.array([1.0, 1.0]))  # Use previous msg_i_to_C
            msg_Xi_0 = 1.0 * msg_j[0] + w[i, j] * msg_j[1]
            msg_Xi_1 = w[i, j] * msg_j[0] + 1.0 * msg_j[1]
            sum_Xi = msg_Xi_0 + msg_Xi_1
            if i == s:
                new_msg_C_to_i[(i, j)][i] = np.array([0.0, 1.0])
            elif i == t:
                new_msg_C_to_i[(i, j)][i] = np.array([1.0, 0.0])
            else:
                if sum_Xi == 0:
                    new_msg_C_to_i[(i, j)][i] = np.array([0.5, 0.5])
                else:
                    new_msg_C_to_i[(i, j)][i] = np.array([msg_Xi_0, msg_Xi_1]) / sum_Xi
            # Message to j
            msg_i = msg_i_to_C[(i, j)].get(i, np.array([1.0, 1.0]))  # Use previous msg_i_to_C
            msg_Xj_0 = 1.0 * msg_i[0] + w[i, j] * msg_i[1]
            msg_Xj_1 = w[i, j] * msg_i[0] + 1.0 * msg_i[1]
            sum_Xj = msg_Xj_0 + msg_Xj_1
            if j == s:
                new_msg_C_to_i[(i, j)][j] = np.array([0.0, 1.0])
            elif j == t:
                new_msg_C_to_i[(i, j)][j] = np.array([1.0, 0.0])
            else:
                if sum_Xj == 0:
                    new_msg_C_to_i[(i, j)][j] = np.array([0.5, 0.5])
                else:
                    new_msg_C_to_i[(i, j)][j] = np.array([msg_Xj_0, msg_Xj_1]) / sum_Xj

        # Compute new variable to cluster messages based on previous cluster to variable messages
        new_msg_i_to_C = {}
        for (i, j) in edges:
            new_msg_i_to_C[(i, j)] = {}
            # Message from i to C (i,j)
            edges_connected_to_i = [e for e in edges if (e[0] == i or e[1] == i) and e != (i, j)]
            product_Xi_0 = 1.0
            product_Xi_1 = 1.0
            for edge in edges_connected_to_i:
                if edge[0] == i:
                    msg = msg_C_to_i[edge].get(i, np.array([0.5, 0.5]))
                else:
                    msg = msg_C_to_i[edge].get(i, np.array([0.5, 0.5]))
                product_Xi_0 *= msg[0]
                product_Xi_1 *= msg[1]
            if i == s:
                product_Xi_0, product_Xi_1 = 0.0, product_Xi_1
            elif i == t:
                product_Xi_0, product_Xi_1 = product_Xi_0, 0.0
            sum_prod_Xi = product_Xi_0 + product_Xi_1
            if sum_prod_Xi == 0:
                new_msg_i_to_C[(i, j)][i] = np.array([0.5, 0.5])
            else:
                new_msg_i_to_C[(i, j)][i] = np.array([product_Xi_0, product_Xi_1]) / sum_prod_Xi

            # Message from j to C (i,j)
            edges_connected_to_j = [e for e in edges if (e[0] == j or e[1] == j) and e != (i, j)]
            product_Xj_0 = 1.0
            product_Xj_1 = 1.0
            for edge in edges_connected_to_j:
                if edge[0] == j:
                    msg = msg_C_to_i[edge].get(j, np.array([0.5, 0.5]))
                else:
                    msg = msg_C_to_i[edge].get(j, np.array([0.5, 0.5]))
                product_Xj_0 *= msg[0]
                product_Xj_1 *= msg[1]
            if j == s:
                product_Xj_0, product_Xj_1 = 0.0, product_Xj_1
            elif j == t:
                product_Xj_0, product_Xj_1 = product_Xj_0, 0.0
            sum_prod_Xj = product_Xj_0 + product_Xj_1
            if sum_prod_Xj == 0:
                new_msg_i_to_C[(i, j)][j] = np.array([0.5, 0.5])
            else:
                new_msg_i_to_C[(i, j)][j] = np.array([product_Xj_0, product_Xj_1]) / sum_prod_Xj

        # Update all messages after computing new ones
        msg_C_to_i = new_msg_C_to_i
        msg_i_to_C = new_msg_i_to_C

    # Compute node beliefs
    nodes_beliefs = calculate_node_beliefs(n, edges, msg_C_to_i)
    # Compute edge beliefs
    edge_beliefs = calculate_edge_beliefs(edges, msg_i_to_C)

    # Compute Bethe free energy
    free_energy = 0.0
    for i in nodes_beliefs:
        belief = nodes_beliefs[i]
        free_energy += np.sum(belief * np.log(belief + 1e-12))

    for (i, j) in edges:
        belief_ij = edge_beliefs[(i, j)]
        for Xi in [0, 1]:
            for Xj in [0, 1]:
                edge_potential = w[i, j] if Xi != Xj else 1.0
                free_energy -= belief_ij[Xi, Xj] * np.log(edge_potential + 1e-12)
                belief_i = nodes_beliefs[i][Xi]
                belief_j = nodes_beliefs[j][Xj]
                if belief_i * belief_j == 0:
                    mutual_info_term = 0
                else:
                    mutual_info_term = belief_ij[Xi, Xj] * np.log((belief_ij[Xi, Xj] + 1e-12) / (belief_i * belief_j + 1e-12))
                free_energy += mutual_info_term

    Z = np.exp(-free_energy)
    return round(Z, 0)

def max_product(M, w, s, t, its):
    n = M.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if M[i, j] != 0 and i < j]

    # Initialize messages from variables to clusters (edges)
    msg_i_to_C = init_msg_i_to_C(edges)

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

if __name__ == "__main__":
    # s (0) ----2---- t (1)
    M = np.array([[0, 1], [1, 0]])
    w = np.array([[0, 2], [2, 0]])
    s, t, its = 0, 1, 1
    print("\nTest Case 1:")
    print("Sum-Product Z:", sum_product(M, w, s, t, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    # s(0) - ---3 - --- (1) - ---4 - --- t(2)
    M = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    w = np.array([[0, 3, 0], [3, 0, 4], [0, 4, 0]])
    s, t, its = 0, 2, 1
    print("\nTest Case 2:")
    print("Sum-Product Z:", sum_product(M, w, s, t, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    M = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])
    w = np.array([
        [0, 2, 3, 0],
        [2, 0, 0, 4],
        [3, 0, 0, 5],
        [0, 4, 5, 0]
    ])
    s, t, its = 0, 3, 1
    print("\nTest Case 3:")
    print("Sum-Product Z:", sum_product(M, w, s, t, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    M = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    w = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]])
    s, t, its = 0, 2, 10
    print("\nTest Case 4:")
    print("Sum-Product Z:", sum_product(M, w, s, t, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    M = np.array([[0, 0], [0, 0]])
    w = np.array([[0, 0], [0, 0]])
    s, t, its = 0, 1, 1
    print("\nTest Case 5:")
    print("Sum-Product Z:", sum_product(M, w, s, t, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))