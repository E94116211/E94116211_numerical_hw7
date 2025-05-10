import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

# 系統矩陣 A 和向量 b
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for it in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    
    for it in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def sor(A, b, omega=1.1, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    
    for it in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Jacobi
x_jacobi = jacobi(A, b)
print("Jacobi method:\n", x_jacobi)

# Gauss-Seidel
x_gs = gauss_seidel(A, b)
print("Gauss-Seidel method:\n", x_gs)

# SOR
x_sor = sor(A, b, omega=1.1)
print("SOR method:\n", x_sor)

# Conjugate Gradient
x_cg, _ = cg(A, b)
print("Conjugate Gradient method:\n", x_cg)
