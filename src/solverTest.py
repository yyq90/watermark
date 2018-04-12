import numpy as np

if __name__ == '__main__':
    A = np.diag(np.arange(1, 5))
    b = np.ones(4)

    # Timing comparison
    from scipy.sparse import rand
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import coo_matrix
    import time
    n = 10000
    i = j = np.arange(n)
    diag = np.ones(n)
    A = rand(n, n, density=0.004)
    A = A.tocsr()
    A[i, j] = diag
    b = np.ones(n)

    t0 = time.time()
    x = spsolve(A, b)
    dt1 = time.time() - t0
    print ("scipy.sparse.linalg.spsolve time: %s" %dt1)
    from scikits.umfpack import spsolve, splu

    t0 = time.time()
    # lu = splu(A)
    x = spsolve(A, b)
    dt1 = time.time() - t0
    print("umfpack solver time: %s" % dt1)

    # from pysparse.sparse import spmatrix
    # from pysparse.direct import superlu
