import numpy as np


def nipals(X, Y, a):
    # [P, Q, W, B] = nipals(X, Y, a)

    # Inputs:
    # X is a matrix of inputs
    # Y is a matrix of outputs
    # a is the number of principal components
    
    # Make a copy of the original X and Y
    X0 = X.copy()
    Y0 = Y.copy()
    
    # Determine the size of the matrices
    n, m = X.shape
    p = Y.shape[1]
    
    tolerance = 1e-6
    
    # Initialize the ouputs
    P = np.zeros(shape = (m, a)) # PC of X 
    Q = np.zeros(shape = (p, a)) # PC of Y
    W = np.zeros(shape = (m, a)) # X weights
    B = np.zeros(shape = (a, a)) # Regression coeff
    T = np.zeros(shape = (n, a)) # Loadings of X
    
    Yres = Y
    
    for h in range(a):
        tOld = np.zeros(n)
        u = Yres[:, 0]
        while True:
            wOld = np.dot(X.T, u)
            w = wOld / np.linalg.norm(wOld)
            t = np.dot(X, w)
            qOld = np.dot(Yres.T, t)
            q = qOld / np.linalg.norm(qOld)
            u = np.dot(Yres, q)
            # Test for convergence
            err = np.linalg.norm(tOld - t) / np.linalg.norm(t)
            if err < tolerance:
                # Calculate the final values for this component
                pOld = np.dot(X.T, t)
                p = pOld / np.linalg.norm(pOld)
                # Store component vectors into the arrays
                P[:, h] = p
                Q[:, h] = q
                W[:, h] = w
                T[:, h] = t
                # One difference from Geladi's algorithm is that
                # we never calculate the X-residual, so we need
                # to calculate the regression coefficients a
                # bit differently. (See Eq. 14 in Geladi.)
                B[:h+1, h] = np.linalg.lstsq(T[:, :h+1], u, rcond=None)[0]
                Ypred = np.dot(np.dot(T, B), Q.T)
                Yres = Y0 - Ypred
                break
            else:
                tOld = t

    return P, Q, W, B