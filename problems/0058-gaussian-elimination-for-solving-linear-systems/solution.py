import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination
    with partial pivoting.
    """

    A = A.astype(float)
    b = b.astype(float)

    n = A.shape[0]

    # -------------------------
    # Forward Elimination
    # -------------------------

    for i in range(n):

        # Partial Pivoting
        row_indices = i + np.argmax(np.abs(A[i:, i]))

        if row_indices != i:
            A[[i, row_indices]] = A[[row_indices, i]]
            b[[i, row_indices]] = b[[row_indices, i]]

        # Elimination below pivot
        for j in range(i + 1, n):

            m = A[j][i] / A[i][i]

            # Update A
            for k in range(i, n):
                A[j][k] = A[j][k] - m * A[i][k]

            # Update b
            b[j] = b[j] - m * b[i]

    # -------------------------
    # Backward Substitution
    # -------------------------

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        s = 0

        for j in range(i + 1, n):
            s += A[i][j] * x[j]

        x[i] = (b[i] - s) / A[i][i]

    return x