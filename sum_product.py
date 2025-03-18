import numpy as np

def init_msg_i_to_C(edges, s, t):
    msg_i_to_C = {}

    for (i, j) in edges:
        msg_i_to_C[(i, j)] = {}

        if i == s:
            msg_i_to_C[(i, j)][i] = np.array([0.0, 1.0])
        elif i == t:
            msg_i_to_C[(i, j)][i] = np.array([1.0, 0.0])
        else:
            msg_i = np.array([1.0, 1.0])
            msg_i /= msg_i.sum()
            msg_i_to_C[(i, j)][i] = msg_i

        if j == s:
            msg_i_to_C[(i, j)][j] = np.array([0.0, 1.0])
        elif j == t:
            msg_i_to_C[(i, j)][j] = np.array([1.0, 0.0])
        else:
            msg_j = np.array([1.0, 1.0])
            msg_j /= msg_j.sum()
            msg_i_to_C[(i, j)][j] = msg_j

    return msg_i_to_C


def init_msg_C_to_i(edges, weights, msg_i_to_C):
    msg_C_to_i = {}

    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}

        msg_j = msg_i_to_C[(i, j)][j]
        msg_Xi_0 = 1.0 * msg_j[0] + weights[i, j] * msg_j[1]
        msg_Xi_1 = weights[i, j] * msg_j[0] + 1.0 * msg_j[1]
        sum_Xi = msg_Xi_0 + msg_Xi_1
        if sum_Xi == 0:
            msg_C_to_i[(i, j)][i] = np.array([0.5, 0.5])
        else:
            msg_C_to_i[(i, j)][i] = np.array([msg_Xi_0, msg_Xi_1]) / sum_Xi

        msg_i = msg_i_to_C[(i, j)][i]
        msg_Xj_0 = 1.0 * msg_i[0] + weights[i, j] * msg_i[1]
        msg_Xj_1 = weights[i, j] * msg_i[0] + 1.0 * msg_i[1]
        sum_Xj = msg_Xj_0 + msg_Xj_1
        if sum_Xj == 0:
            msg_C_to_i[(i, j)][j] = np.array([0.5, 0.5])
        else:
            msg_C_to_i[(i, j)][j] = np.array([msg_Xj_0, msg_Xj_1]) / sum_Xj

    return msg_C_to_i

def updt_msg_C_to_i(edges, weights, s, t, msg_i_to_C):
    msg_C_to_i = {}
    for (i, j) in edges:
        msg_C_to_i[(i, j)] = {}

        msg_j = msg_i_to_C[(i, j)].get(j, np.array([1.0, 1.0]))
        msg_Xi_0 = 1.0 * msg_j[0] + weights[i, j] * msg_j[1]
        msg_Xi_1 = weights[i, j] * msg_j[0] + 1.0 * msg_j[1]
        sum_Xi = msg_Xi_0 + msg_Xi_1
        if i == s:
            msg_C_to_i[(i, j)][i] = np.array([0.0, 1.0])
        elif i == t:
            msg_C_to_i[(i, j)][i] = np.array([1.0, 0.0])
        else:
            if sum_Xi == 0:
                msg_C_to_i[(i, j)][i] = np.array([0.5, 0.5])
            else:
                msg_C_to_i[(i, j)][i] = np.array([msg_Xi_0, msg_Xi_1]) / sum_Xi

        msg_i = msg_i_to_C[(i, j)].get(i, np.array([1.0, 1.0]))
        msg_Xj_0 = 1.0 * msg_i[0] + weights[i, j] * msg_i[1]
        msg_Xj_1 = weights[i, j] * msg_i[0] + 1.0 * msg_i[1]
        sum_Xj = msg_Xj_0 + msg_Xj_1
        if j == s:
            msg_C_to_i[(i, j)][j] = np.array([0.0, 1.0])
        elif j == t:
            msg_C_to_i[(i, j)][j] = np.array([1.0, 0.0])
        else:
            if sum_Xj == 0:
                msg_C_to_i[(i, j)][j] = np.array([0.5, 0.5])
            else:
                msg_C_to_i[(i, j)][j] = np.array([msg_Xj_0, msg_Xj_1]) / sum_Xj
    return msg_C_to_i

def updt_msg_i_to_C(edges, s, t, msg_C_to_i):
    # Compute new variable to cluster messages based on previous cluster to variable messages
    msg_i_to_C = {}
    for (i, j) in edges:
        msg_i_to_C[(i, j)] = {}
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
            msg_i_to_C[(i, j)][i] = np.array([0.5, 0.5])
        else:
            msg_i_to_C[(i, j)][i] = np.array([product_Xi_0, product_Xi_1]) / sum_prod_Xi

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
            msg_i_to_C[(i, j)][j] = np.array([0.5, 0.5])
        else:
            msg_i_to_C[(i, j)][j] = np.array([product_Xj_0, product_Xj_1]) / sum_prod_Xj
    return msg_i_to_C

def calculate_node_beliefs(n, edges, s, t, msg_C_to_i):
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

def calculate_edge_beliefs(edges, weights, msg_i_to_C):
    edge_beliefs = {}
    for (i, j) in edges:
        belief = np.zeros((2, 2))
        for Xi in [0, 1]:
            for Xj in [0, 1]:
                edge_potential = weights[i, j] if Xi != Xj else 1.0
                msg_i = msg_i_to_C[(i, j)].get(i, np.array([0.5, 0.5]))[Xi]
                msg_j = msg_i_to_C[(i, j)].get(j, np.array([0.5, 0.5]))[Xj]
                belief[Xi, Xj] = edge_potential * msg_i * msg_j
        sum_belief = belief.sum()
        if sum_belief == 0:
            edge_beliefs[(i, j)] = np.ones((2, 2)) / 4
        else:
            edge_beliefs[(i, j)] = belief / sum_belief
    return edge_beliefs

def calculate_beth_free_energy(nodes_beliefs, edge_beliefs, weights, edges):
    free_energy = 0.0
    for i in nodes_beliefs:
        belief = nodes_beliefs[i]
        free_energy += np.sum(belief * np.log(belief + 1e-12))

    for (i, j) in edges:
        belief_ij = edge_beliefs[(i, j)]
        for Xi in [0, 1]:
            for Xj in [0, 1]:
                edge_potential = weights[i, j] if Xi != Xj else 1.0
                free_energy -= belief_ij[Xi, Xj] * np.log(edge_potential + 1e-12)
                belief_i = nodes_beliefs[i][Xi]
                belief_j = nodes_beliefs[j][Xj]
                if belief_i * belief_j == 0:
                    mutual_info_term = 0
                else:
                    mutual_info_term = belief_ij[Xi, Xj] * np.log(
                        (belief_ij[Xi, Xj] + 1e-12) / (belief_i * belief_j + 1e-12))
                free_energy += mutual_info_term
    return free_energy

def sum_product(adjacency_matrix, weights, s, t, its):
    n = adjacency_matrix.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if adjacency_matrix[i, j] != 0 and i < j]

    msg_i_to_C = init_msg_i_to_C(edges, s, t)
    msg_C_to_i = init_msg_C_to_i(edges, weights, msg_i_to_C)

    for _ in range(its):
        msg_C_to_i = updt_msg_C_to_i(edges, weights, s, t, msg_i_to_C)
        msg_i_to_C = updt_msg_i_to_C(edges, s, t, msg_C_to_i)

    nodes_beliefs = calculate_node_beliefs(n, edges, s, t, msg_C_to_i)
    edge_beliefs = calculate_edge_beliefs(edges, weights, msg_i_to_C)
    Z = np.exp(-calculate_beth_free_energy(nodes_beliefs, edge_beliefs, weights, edges))
    return round(Z, 0)