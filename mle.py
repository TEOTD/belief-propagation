import numpy as np
import pandas as pd

from gibbs_sampling import gibbs
from loopy_bpe import sumprod


def learnstcuts(A, s, t, samples, bp_its=1000, lr=1.0, max_iter=1000, decay=0.95, reg=1e-5, momentum=0.9, initial_guess=1.0):
    n = A.shape[0]
    m = samples.shape[1]

    # Compute empirical edge marginals
    E_emp = np.zeros_like(A, dtype=float)
    edges = [(i, j) for i in range(n) for j in range(n) if A[i, j] and i < j]
    for i, j in edges:
        count = np.sum(samples[i, :] != samples[j, :])
        E_emp[i, j] = count / m
        E_emp[j, i] = E_emp[i, j]

    theta = np.ones_like(A, dtype=float) * initial_guess
    prev_update = np.zeros_like(theta)

    # Gradient ascent with momentum and L2 regularization
    for iter in range(max_iter):
        current_lr = lr * (decay ** iter)
        w = np.exp(theta)  # Convert to weights

        # Compute model expectations
        Z, beliefs = sumprod(A, s, t, w, its=bp_its)
        E_model = np.zeros_like(A, dtype=float)
        for (i, j) in edges:
            prob_01 = beliefs.get(((i, j), (0, 1)), 0)
            prob_10 = beliefs.get(((i, j), (1, 0)), 0)
            E_model[i, j] = prob_01 + prob_10
            E_model[j, i] = E_model[i, j]

        # Gradient = (E_emp - E_model) - λθ (L2 regularization)
        gradient = (E_emp - E_model) - reg * theta

        # Momentum update
        update = momentum * prev_update + current_lr * gradient
        theta += update
        prev_update = update

        # Clip parameters to prevent overflow
        theta = np.clip(theta, -10, 10)

    # Final symmetric weights
    w = np.exp(theta)
    for (i, j) in edges:
        w[j, i] = w[i, j]

    return w

if __name__ == "__main__":
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    true_w = np.array([[0, np.exp(1), 0], [np.exp(1), 0, np.exp(1)], [0, np.exp(1), 0]])
    s, t = 0, 2

    # Sample sizes to test
    sample_sizes = [10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    results = []
    z_true, _ = sumprod(A, s, t, true_w, its=1000)
    for m in sample_sizes:
        # Generate samples using Gibbs
        _, samples = gibbs(A, s=s, t=t, w=true_w, burnin=1000, its=m)
        # Learn weights
        learned_w = learnstcuts(A, s, t, samples)
        # Compute estimated Z
        Z_estimated, _ = sumprod(A, s, t, learned_w, its=1000)
        results.append(Z_estimated)

    df = pd.DataFrame({
        "Samples": sample_sizes,
        "Estimated Z": np.round(results, 4),
        "True Z": np.round(z_true, 4)
    })
    print("\nEstimated Partition Function vs. True Z:")
    print(df.to_markdown(index=False))
