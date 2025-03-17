import numpy as np


def sum_product(M, w, s, t, its):
    n = M.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if M[i, j] != 0 and i < j]
    # Initialize messages
    m_v2f = {}
    m_f2v = {}

    for (i, j) in edges:
        m_v2f[(i, j)] = {}
        # Message from i to (i,j)
        if i == s:
            m_v2f[(i, j)][i] = np.array([0.0, 1.0])
        elif i == t:
            m_v2f[(i, j)][i] = np.array([1.0, 0.0])
        else:
            m_v2f[(i, j)][i] = np.array([1.0, 1.0])
            m_v2f[(i, j)][i] /= m_v2f[(i, j)][i].sum()
        # Message from j to (i,j)
        if j == s:
            m_v2f[(i, j)][j] = np.array([0.0, 1.0])
        elif j == t:
            m_v2f[(i, j)][j] = np.array([1.0, 0.0])
        else:
            m_v2f[(i, j)][j] = np.array([1.0, 1.0])
            m_v2f[(i, j)][j] /= m_v2f[(i, j)][j].sum()

    for (i, j) in edges:
        m_f2v[(i, j)] = {}
        # Message from factor (i,j) to i
        msg_j = m_v2f.get((i, j), {}).get(j, np.array([1.0, 1.0]))
        m_to_i_0 = 1.0 * msg_j[0] + w[i, j] * msg_j[1]
        m_to_i_1 = w[i, j] * msg_j[0] + 1.0 * msg_j[1]
        sum_m = m_to_i_0 + m_to_i_1
        if sum_m == 0:
            m_f2v[(i, j)][i] = np.array([0.5, 0.5])
        else:
            m_f2v[(i, j)][i] = np.array([m_to_i_0, m_to_i_1]) / sum_m
        # Message from factor (i,j) to j
        msg_i = m_v2f.get((i, j), {}).get(i, np.array([1.0, 1.0]))
        m_to_j_0 = 1.0 * msg_i[0] + w[i, j] * msg_i[1]
        m_to_j_1 = w[i, j] * msg_i[0] + 1.0 * msg_i[1]
        sum_mj = m_to_j_0 + m_to_j_1
        if sum_mj == 0:
            m_f2v[(i, j)][j] = np.array([0.5, 0.5])
        else:
            m_f2v[(i, j)][j] = np.array([m_to_j_0, m_to_j_1]) / sum_mj

    for _ in range(its):
        new_m_f2v = {}
        new_m_v2f = {}

        # Update factor to variable messages
        for (i, j) in edges:
            new_m_f2v[(i, j)] = {}
            # Message to i
            msg_j = m_v2f.get((i, j), {}).get(j, np.array([1.0, 1.0]))
            m_to_i_0 = 1.0 * msg_j[0] + w[i, j] * msg_j[1]
            m_to_i_1 = w[i, j] * msg_j[0] + 1.0 * msg_j[1]
            sum_m = m_to_i_0 + m_to_i_1
            if sum_m == 0:
                new_m_f2v[(i, j)][i] = np.array([0.5, 0.5])
            else:
                new_m_f2v[(i, j)][i] = np.array([m_to_i_0, m_to_i_1]) / sum_m
            # Message to j
            msg_i = m_v2f.get((i, j), {}).get(i, np.array([1.0, 1.0]))
            m_to_j_0 = 1.0 * msg_i[0] + w[i, j] * msg_i[1]
            m_to_j_1 = w[i, j] * msg_i[0] + 1.0 * msg_i[1]
            sum_mj = m_to_j_0 + m_to_j_1
            if sum_mj == 0:
                new_m_f2v[(i, j)][j] = np.array([0.5, 0.5])
            else:
                new_m_f2v[(i, j)][j] = np.array([m_to_j_0, m_to_j_1]) / sum_mj

        # Update variable to factor messages
        for (i, j) in edges:
            new_m_v2f[(i, j)] = {}
            # Message from i to (i,j)
            connected_edges_i = [e for e in edges if (e[0] == i or e[1] == i) and e != (i, j)]
            product_0 = 1.0
            product_1 = 1.0
            for e in connected_edges_i:
                if e[0] == i:
                    msg = new_m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
                else:
                    msg = new_m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
                product_0 *= msg[0]
                product_1 *= msg[1]
            if i == s:
                product_0, product_1 = 0.0, product_1
            elif i == t:
                product_0, product_1 = product_0, 0.0
            sum_prod = product_0 + product_1
            if sum_prod == 0:
                new_m_v2f[(i, j)][i] = np.array([0.5, 0.5])
            else:
                new_m_v2f[(i, j)][i] = np.array([product_0, product_1]) / sum_prod
            # Message from j to (i,j)
            connected_edges_j = [e for e in edges if (e[0] == j or e[1] == j) and e != (i, j)]
            product_j_0 = 1.0
            product_j_1 = 1.0
            for e in connected_edges_j:
                if e[0] == j:
                    msg = new_m_f2v.get(e, {}).get(j, np.array([1.0, 1.0]))
                else:
                    msg = new_m_f2v.get(e, {}).get(j, np.array([1.0, 1.0]))
                product_j_0 *= msg[0]
                product_j_1 *= msg[1]
            if j == s:
                product_j_0, product_j_1 = 0.0, product_j_1
            elif j == t:
                product_j_0, product_j_1 = product_j_0, 0.0
            sum_prod_j = product_j_0 + product_j_1
            if sum_prod_j == 0:
                new_m_v2f[(i, j)][j] = np.array([0.5, 0.5])
            else:
                new_m_v2f[(i, j)][j] = np.array([product_j_0, product_j_1]) / sum_prod_j

        m_f2v = new_m_f2v
        m_v2f = new_m_v2f

    # Compute beliefs
    var_beliefs = {}
    for i in range(n):
        connected_edges = [e for e in edges if e[0] == i or e[1] == i]
        product = np.array([1.0, 1.0])
        for e in connected_edges:
            if e[0] == i:
                msg = m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
            else:
                msg = m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
            product *= msg
        if i == s:
            product = np.array([0.0, product[1]])
        elif i == t:
            product = np.array([product[0], 0.0])
        product /= product.sum()
        var_beliefs[i] = product

    edge_beliefs = {}
    for (i, j) in edges:
        belief = np.zeros((2, 2))
        for xi in [0, 1]:
            for xj in [0, 1]:
                psi = w[i, j] if xi != xj else 1.0
                mi = m_v2f.get((i, j), {}).get(i, np.array([1.0, 1.0]))[xi]
                mj = m_v2f.get((i, j), {}).get(j, np.array([1.0, 1.0]))[xj]
                belief[xi, xj] = psi * mi * mj
        belief /= belief.sum()
        edge_beliefs[(i, j)] = belief

    # Compute Bethe free energy
    # Compute Bethe free energy
    F = 0.0
    for i in var_beliefs:
        b = var_beliefs[i]
        F += np.sum(b * np.log(b + 1e-12))  # Node entropies

    for (i, j) in edges:
        b_ij = edge_beliefs[(i, j)]
        for xi in [0, 1]:
            for xj in [0, 1]:
                psi_val = w[i, j] if xi != xj else 1.0
                # Energy term (subtract)
                F -= b_ij[xi, xj] * np.log(psi_val + 1e-12)
                bi = var_beliefs[i][xi]
                bj = var_beliefs[j][xj]
                if bi * bj == 0:
                    term = 0
                else:
                    term = b_ij[xi, xj] * np.log((b_ij[xi, xj] + 1e-12) / (bi * bj + 1e-12))
                # Mutual information term (add)
                F += term  # Corrected from F -= term

    Z = np.exp(-F)
    return round(Z, 0)


def max_product(M, w, s, t, its):
    n = M.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if M[i, j] != 0 and i != j]

    # Initialize messages
    m_v2f = {}
    m_f2v = {}

    for (i, j) in edges:
        m_v2f[(i, j)] = {}
        # Message from i to (i,j)
        if i == s:
            m_v2f[(i, j)][i] = np.array([0.0, 1.0])
        elif i == t:
            m_v2f[(i, j)][i] = np.array([1.0, 0.0])
        else:
            m_v2f[(i, j)][i] = np.array([1.0, 1.0])
            m_v2f[(i, j)][i] /= m_v2f[(i, j)][i].sum()
        # Message from j to (i,j)
        if j == s:
            m_v2f[(i, j)][j] = np.array([0.0, 1.0])
        elif j == t:
            m_v2f[(i, j)][j] = np.array([1.0, 0.0])
        else:
            m_v2f[(i, j)][j] = np.array([1.0, 1.0])
            m_v2f[(i, j)][j] /= m_v2f[(i, j)][j].sum()

    for (i, j) in edges:
        m_f2v[(i, j)] = {}
        # Message from factor (i,j) to i
        msg_j = m_v2f.get((i, j), {}).get(j, np.array([1.0, 1.0]))
        m_to_i_0 = 1.0 * msg_j[0] + w[i, j] * msg_j[1]
        m_to_i_1 = w[i, j] * msg_j[0] + 1.0 * msg_j[1]
        max_val = max(m_to_i_0, m_to_i_1)
        if max_val == 0:
            m_f2v[(i, j)][i] = np.array([1.0, 1.0])
        else:
            m_f2v[(i, j)][i] = np.array([m_to_i_0, m_to_i_1]) / max_val
        # Message from factor (i,j) to j
        msg_i = m_v2f.get((i, j), {}).get(i, np.array([1.0, 1.0]))
        m_to_j_0 = 1.0 * msg_i[0] + w[i, j] * msg_i[1]
        m_to_j_1 = w[i, j] * msg_i[0] + 1.0 * msg_i[1]
        max_val_j = max(m_to_j_0, m_to_j_1)
        if max_val_j == 0:
            m_f2v[(i, j)][j] = np.array([1.0, 1.0])
        else:
            m_f2v[(i, j)][j] = np.array([m_to_j_0, m_to_j_1]) / max_val_j

    for _ in range(its):
        new_m_f2v = {}
        new_m_v2f = {}

        # Update factor to variable messages
        for (i, j) in edges:
            new_m_f2v[(i, j)] = {}
            # Message to i
            msg_j = m_v2f.get((i, j), {}).get(j, np.array([1.0, 1.0]))
            m_to_i_0 = 1.0 * msg_j[0] + w[i, j] * msg_j[1]
            m_to_i_1 = w[i, j] * msg_j[0] + 1.0 * msg_j[1]
            max_val = max(m_to_i_0, m_to_i_1)
            if max_val == 0:
                new_m_f2v[(i, j)][i] = np.array([1.0, 1.0])
            else:
                new_m_f2v[(i, j)][i] = np.array([m_to_i_0, m_to_i_1]) / max_val
            # Message to j
            msg_i = m_v2f.get((i, j), {}).get(i, np.array([1.0, 1.0]))
            m_to_j_0 = 1.0 * msg_i[0] + w[i, j] * msg_i[1]
            m_to_j_1 = w[i, j] * msg_i[0] + 1.0 * msg_i[1]
            max_val_j = max(m_to_j_0, m_to_j_1)
            if max_val_j == 0:
                new_m_f2v[(i, j)][j] = np.array([1.0, 1.0])
            else:
                new_m_f2v[(i, j)][j] = np.array([m_to_j_0, m_to_j_1]) / max_val_j

        # Update variable to factor messages
        for (i, j) in edges:
            new_m_v2f[(i, j)] = {}
            # Message from i to (i,j)
            connected_edges_i = [e for e in edges if (e[0] == i or e[1] == i) and e != (i, j)]
            product_0 = 1.0
            product_1 = 1.0
            for e in connected_edges_i:
                if e[0] == i:
                    msg = new_m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
                else:
                    msg = new_m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
                product_0 *= msg[0]
                product_1 *= msg[1]
            if i == s:
                product_0, product_1 = 0.0, product_1
            elif i == t:
                product_0, product_1 = product_0, 0.0
            max_val_i = max(product_0, product_1)
            if max_val_i == 0:
                new_m_v2f[(i, j)][i] = np.array([1.0, 1.0])
            else:
                new_m_v2f[(i, j)][i] = np.array([product_0, product_1]) / max_val_i
            # Message from j to (i,j)
            connected_edges_j = [e for e in edges if (e[0] == j or e[1] == j) and e != (i, j)]
            product_j_0 = 1.0
            product_j_1 = 1.0
            for e in connected_edges_j:
                if e[0] == j:
                    msg = new_m_f2v.get(e, {}).get(j, np.array([1.0, 1.0]))
                else:
                    msg = new_m_f2v.get(e, {}).get(j, np.array([1.0, 1.0]))
                product_j_0 *= msg[0]
                product_j_1 *= msg[1]
            if j == s:
                product_j_0, product_j_1 = 0.0, product_j_1
            elif j == t:
                product_j_0, product_j_1 = product_j_0, 0.0
            max_val_j = max(product_j_0, product_j_1)
            if max_val_j == 0:
                new_m_v2f[(i, j)][j] = np.array([1.0, 1.0])
            else:
                new_m_v2f[(i, j)][j] = np.array([product_j_0, product_j_1]) / max_val_j

        m_f2v = new_m_f2v
        m_v2f = new_m_v2f

    # Compute max-marginals
    var_beliefs = {}
    for i in range(n):
        connected_edges = [e for e in edges if e[0] == i or e[1] == i]
        product = np.array([1.0, 1.0])
        for e in connected_edges:
            if e[0] == i:
                msg = m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
            else:
                msg = m_f2v.get(e, {}).get(i, np.array([1.0, 1.0]))
            product *= msg
        if i == s:
            product = np.array([0.0, product[1]])
        elif i == t:
            product = np.array([product[0], 0.0])
        product /= product.sum()
        var_beliefs[i] = product

    assignment = np.zeros(n)
    assignment[s] = 1.0
    assignment[t] = 0.0
    for i in range(n):
        if i == s or i == t:
            continue
        b0 = var_beliefs[i][0]
        b1 = var_beliefs[i][1]
        if b0 > b1:
            assignment[i] = 0.0
        elif b1 > b0:
            assignment[i] = 1.0
        else:
            assignment[i] = 0.5

    return assignment



if __name__ == "__main__":
    # Test Case 1: Single Edge (Tree)
    M1 = np.array([[0, 1], [1, 0]])
    w1 = np.array([[0, 2], [2, 0]])
    s1, t1 = 0, 1
    print("Test Case 1:")
    print("Sum-Product Z:", sum_product(M1, w1, s1, t1, 10))
    # print("Max-Product Assignment:", max_product(M1, w1, s1, t1, 10))

    # Test Case 2: Chain (Tree)
    M2 = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    w2 = np.zeros((4, 4))
    w2[0, 1] = 2
    w2[1, 0] = 2
    w2[1, 2] = 3
    w2[2, 1] = 3
    w2[2, 3] = 4
    w2[3, 2] = 4
    s2, t2 = 0, 3
    print("\nTest Case 2:")
    print("Sum-Product Z:", sum_product(M2, w2, s2, t2, 10))
    # print("Max-Product Assignment:", max_product(M2, w2, s2, t2, 10))

    # Test Case 3: Disconnected Graph
    M3 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    w3 = np.zeros((4, 4))
    w3[2, 3] = 1
    w3[3, 2] = 1
    s3, t3 = 0, 3
    print("\nTest Case 3:")
    print("Sum-Product Z:", sum_product(M3, w3, s3, t3, 10))
    # print("Max-Product Assignment:", max_product(M3, w3, s3, t3, 10))

    # Test Case 4: Cyclic Graph (Triangle)
    M4 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    w4 = np.zeros((3, 3))
    w4[0, 1] = 2
    w4[1, 0] = 2
    w4[1, 2] = 3
    w4[2, 1] = 3
    w4[0, 2] = 4
    w4[2, 0] = 4
    s4, t4 = 0, 2
    print("\nTest Case 4:")
    print("Sum-Product Z:", sum_product(M4, w4, s4, t4, 10))
    # print("Max-Product Assignment:", max_product(M4, w4, s4, t4, 10))

    # Test Case 5: All Edges Zero
    M5 = np.array([[0, 1], [1, 0]])
    w5 = np.zeros((2, 2))
    s5, t5 = 0, 1
    print("\nTest Case 5:")
    print("Sum-Product Z:", sum_product(M5, w5, s5, t5, 10))
    # print("Max-Product Assignment:", max_product(M5, w5, s5, t5, 10))

    # Test Case 6: Simple Chain Tree
    M6 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    w6 = np.array([
        [0, 2, 0],
        [2, 0, 3],
        [0, 3, 0]
    ])
    s6, t6 = 0, 2
    print("\nTest Case 5:")
    print("Sum-Product Z:", sum_product(M6, w6, s6, t6, 10))
    # print("Max-Product Assignment:", max_product(M6, w6, s6, t6, 10))
