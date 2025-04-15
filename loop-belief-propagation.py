import numpy as np

from max_product import max_product
from sum_product import sum_product

if __name__ == "__main__":
    # s (0) ----2---- t (1)
    M = np.array([[0, 1], [1, 0]])
    w = np.array([[0, 2], [2, 0]])
    s, t, its = 0, 1, 1
    print("\nTest Case 1:")
    print("Sum-Product Z:", sum_product(M, s, t, w, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    # s(0) - ---3 - --- (1) - ---4 - --- t(2)
    M = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    w = np.array([[0, 3, 0], [3, 0, 4], [0, 4, 0]])
    s, t, its = 0, 2, 1
    print("\nTest Case 2:")
    print("Sum-Product Z:", sum_product(M, s, t, w, its))
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
    print("Sum-Product Z:", sum_product(M, s, t, w, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    M = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    w = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]])
    s, t, its = 0, 2, 10
    print("\nTest Case 4:")
    print("Sum-Product Z:", sum_product(M, s, t, w, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    M = np.array([[0, 0], [0, 0]])
    w = np.array([[0, 0], [0, 0]])
    s, t, its = 0, 1, 1
    print("\nTest Case 5:")
    print("Sum-Product Z:", sum_product(M, s, t, w, its))
    print("Max-Product Assignment:", max_product(M, w, s, t, its))

    n = 2
    M_test = np.zeros((n, n), dtype=int)
    for (i, j) in [(0, 1), (1, 0)]:
        M_test[i, j] = 1
        M_test[j, i] = 1
    w_test = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if M_test[i, j] == 1:
                w_test[i, j] = 2.0

    s_idx = 0
    t_idx = 1

    Z_approx = sum_product(M_test, s_idx, t_idx, w_test, its=3)
    print("Approx. partition function =", Z_approx)

    x_map = max_product(M_test, w_test, s_idx, t_idx, its=3)
    print("Max-product assignment =", x_map)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    # Adjacency for a "star": 0 connected to 1,2,3
    n = 4
    M_test = np.zeros((n, n), dtype=int)
    for (i, j) in [(0, 1), (0, 2), (0, 3)]:
        M_test[i, j] = 1
        M_test[j, i] = 1

    # Let's do w=2 for edges (0,1) and (0,2), and w=3 for (0,3)
    w_test = np.zeros((n, n))
    edges = [(0, 1, 2.0), (0, 2, 2.0), (0, 3, 3.0)]
    for (i, j, wval) in edges:
        w_test[i, j] = wval
        w_test[j, i] = wval

    s_idx = 0
    t_idx = 3

    Z_approx = sum_product(M_test, s_idx, t_idx, w_test, its=6)
    print("Tree partition function =", Z_approx)

    x_map = max_product(M_test, w_test, s_idx, t_idx, its=6)
    print("Tree max-product assignment =", x_map)