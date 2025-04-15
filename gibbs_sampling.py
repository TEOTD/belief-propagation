import numpy as np
import pandas as pd

def gibbs(A, s, t, w, burnin, its):
    """
    Gibbs sampling on MRF represented by adjacency matrix A and weight matrix w.

    Parameters:
    A : np.ndarray
        Adjacency matrix of the graph (n x n), where A[other_edge,neighbor] != 0 indicates an edge between other_edge and neighbor.
    s : int
        Index of the source node (fixed to 1).
    t : int
        Index of the sink node (fixed to 0).
    w : np.ndarray
        Edge weight matrix (same shape as A).
    burnin : int
        Number of iterations for burn-in period (samples discarded).
    its : int
        Number of samples to collect after burn-in.

    Returns:
    marginals : np.ndarray
        Estimated marginal probabilities of each node being 1.
    samples : np.ndarray
        Matrix of sampled states, shape (n x its).
    """
    n = A.shape[0]  # Total number of nodes in the graph

    # List of all nodes except the fixed source (s) and sink (t)
    other_edges = [i for i in range(n) if i != s and i != t]

    # Initialize a random binary state vector of length n
    x = np.random.randint(0, 2, n)
    x[s] = 1  # Fix source node to 1
    x[t] = 0  # Fix sink node to 0

    # Initialize marginals accumulator and list to store samples
    marginals = np.zeros(n)
    samples = []

    # Total number of Gibbs sampling iterations (burn-in + actual sampling)
    for iteration in range(burnin + its):
        # Update each node (excluding s and t) one at a time
        for other_edge in other_edges:
            neighbors = np.where(A[other_edge] != 0)[0]  # Get neighbors of node other_edge
            prob_x_1 = 1.0  # Unnormalized probability for x[other_edge] = 1
            prob_x_0 = 1.0  # Unnormalized probability for x[other_edge] = 0

            # Compute conditional probabilities based on neighbors
            for neighbor in neighbors:
                if neighbor == other_edge:
                    continue  # Skip self (shouldn't be in neighbors)

                # If neighbor is off (0), it favors x[other_edge] = 1
                if x[neighbor] == 0:
                    prob_x_1 *= w[other_edge, neighbor]

                # If neighbor is on (1), it favors x[other_edge] = 0
                if x[neighbor] == 1:
                    prob_x_0 *= w[other_edge, neighbor]

            # Normalize the probabilities
            total = prob_x_1 + prob_x_0
            if total == 0:
                prob = 0.5  # If both are zero, assign uniform probability
            else:
                prob = prob_x_1 / total  # Probability of setting x[other_edge] = 1

            # Sample x[other_edge] based on computed probability
            x[other_edge] = 1 if np.random.rand() < prob else 0

        # After burn-in period, record sample
        if iteration >= burnin:
            marginals += x  # Accumulate marginal probabilities
            samples.append(x.copy())  # Store a copy of the current state

    # Average marginals over number of samples collected
    marginals /= its

    # Return marginals and transpose of samples (n x its)
    return marginals, np.array(samples).T


if __name__ == "__main__":
    # ===== GRAPH SETUP =====
    # Nodes: s(0), a(1), b(2), c(3), d(4), e(5), f(6), t(7)
    n = 8
    A = np.zeros((n, n), dtype=int)
    edges = [
        (0, 1), (0, 2), (0, 3),   # s connected to a, b, c
        (1, 0), (1, 3), (1, 4),   # a connected to s, c, d
        (2, 0), (2, 3),           # b connected to s, c
        (3, 0), (3, 1), (3, 2), (3, 5),  # c connected to s, a, b, e
        (4, 1), (4, 6), (4, 7),   # d connected to a, f, t
        (5, 3), (5, 6), (5, 7),   # e connected to c, f, t
        (6, 4), (6, 5), (6, 7),   # f connected to d, e, t
        (7, 4), (7, 5), (7, 6)    # t connected to d, e, f
    ]
    for i, j in edges:
        A[i, j] = 1

    # Weight matrix (all edges have weight 0.5)
    w = np.zeros((n, n))
    w[A == 1] = 0.5

    # ===== GIBBS SAMPLING FOR PART B =====
    params = [2**6, 2**10, 2**14, 2**18]
    table = pd.DataFrame(index=params, columns=params, dtype=float)

    for burnin in params:
        for its in params:
            print(f"Running: burnin={burnin}, its={its}...")
            marginals, _ = gibbs(A, s=0, t=7, w=w, burnin=burnin, its=its)
            table.loc[burnin, its] = np.round(marginals[5], 4)  # Node e is index 5

    # ===== PRINT RESULTS =====
    print("\nEstimated Marginal of Node e:")
    print(table.to_markdown(floatfmt=".4f"))