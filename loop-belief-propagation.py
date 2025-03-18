import numpy as np

from max_product import max_product
from sum_product import sum_product

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