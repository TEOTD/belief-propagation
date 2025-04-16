from collections import defaultdict

import numpy as np
import pandas as pd

from gibbs_sampling import gibbs
from loopy_bpe import sumprod


def learnstcuts(A, s, t, samples, bp_its=1000, lr=1.0, max_iter=1000):
    n = A.shape[0]
    m = samples.shape[1]

    E_emp = np.zeros_like(A, dtype=float)
    edges = [(i, j) for i in range(n) for j in range(n) if A[i, j] and i < j]
    for i, j in edges:
        count = np.sum(samples[i, :] != samples[j, :])
        E_emp[i, j] = count / m
        E_emp[j, i] = E_emp[i, j]

    theta = defaultdict(float)
    for edge in edges:
        theta[edge] = 1.0

    for _ in range(max_iter):
        w = np.zeros((n, n))
        for (i, j) in edges:
            w_ij = np.exp(theta[(i, j)])
            w[i][j] = w_ij
            w[j][i] = w_ij

        Z, beliefs = sumprod(A, s, t, w, its=bp_its)
        E_model = np.zeros_like(A, dtype=float)
        for (i, j) in edges:
            prob_01 = beliefs.get(((i, j), (0, 1)), 0)
            prob_10 = beliefs.get(((i, j), (1, 0)), 0)
            E_model[i, j] = prob_01 + prob_10
            E_model[j, i] = E_model[i, j]

        for edge in edges:
            grad = E_emp[edge] - E_model[edge]
            delta = lr * grad
            theta[edge] += delta

    w = np.zeros((n, n))
    for (i, j) in edges:
        w_ij = np.exp(theta[(i, j)])
        w[i][j] = w_ij
        w[j][i] = w_ij

    return w

if __name__ == "__main__":
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    true_w = np.array([[0, np.exp(1), 0], [np.exp(1), 0, np.exp(1)], [0, np.exp(1), 0]])
    s, t = 0, 2

    sample_sizes = [10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    results = []
    z_true, _ = sumprod(A, s, t, true_w, its=1000)
    for m in sample_sizes:
        _, samples = gibbs(A, s=s, t=t, w=true_w, burnin=1000, its=m)
        learned_w = learnstcuts(A, s, t, samples)
        Z_estimated, _ = sumprod(A, s, t, learned_w, its=1000)
        results.append(Z_estimated)

    df = pd.DataFrame({
        "Samples": sample_sizes,
        "Estimated Z": np.round(results, 4),
        "True Z": np.round(z_true, 4)
    })
    print("\nEstimated Partition Function vs. True Z:")
    print(df.to_markdown(index=False))