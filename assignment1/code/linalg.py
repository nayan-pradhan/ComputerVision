import numpy as np

def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """

    out = None
    ### YOUR CODE HERE

    # # Check whether dot product is possible or not
    # a_col = a.shape[1]
    # b_row = b.shape[0]

    # if (a_col == b_row):
    #     b = b.T # Transverse
    # else:
    #     return -1 # if a_col != b_row
    # # Compute dot product
    # out = sum(sum([i*j for (i, j) in zip(a, b)]))

    out = np.dot(a, b)
    pass

    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    out = None
    ### YOUR CODE HERE

    temp1 = dot_product(a,b)
    temp2 = my_multiplication(M, a.T)
    temp3 = temp1 * temp2.T
    out = temp3.T
    
    pass
    ### END YOUR CODE

    return out

def my_multiplication(a,b):
    res = np.array()
    for r in range(a.shape[1]):
        for c in range(b.shape[0]):
            res[r][c] = dot_product(a[r,:], b[:,c])
    return res

def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u = None
    s = None
    v = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return u, s, v


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    singular_values = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return singular_values


def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return w, v


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return eigenvalues, eigenvectors
