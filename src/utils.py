"""
src/utils.py - Shared model-agnostic numerical and matrix utility functions.

Provides:
- Z-score standardization (mean 0, variance 1)
- Sample covariance computation: S = (1/n) X^T X
- Positive-definiteness & eigenvalue checks (Theta > 0)
- Matrix symmetry verification and symmetrization
- Matrix log-determinant calculation via slogdet
- Sparsity metrics and edge counting for precision matrices
"""

import numpy as np


def standardize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Standardize data matrix X column-wise to zero mean and unit variance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix (patient samples x genes).
    eps : float, default=1e-12
        Small epsilon to prevent division by zero for constant columns.

    Returns
    -------
    X_std : np.ndarray of shape (n_samples, n_features)
        Standardized data with mean 0 and std 1 per column.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Guard against constant features
    std_safe = np.where(std < eps, 1.0, std)
    return (X - mean) / std_safe


def compute_covariance(X: np.ndarray, assume_centered: bool = True) -> np.ndarray:
    """
    Compute the sample covariance matrix S = (1/n) X^T X.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.
    assume_centered : bool, default=True
        If True, assumes X is already centered (zero mean).
        If False, centers X before computing covariance.

    Returns
    -------
    S : np.ndarray of shape (n_features, n_features)
        Sample covariance matrix, guaranteed symmetric.
    """
    X_mat = np.asarray(X, dtype=np.float64)
    n = X_mat.shape[0]
    if n <= 1:
        raise ValueError(f"Need at least 2 samples to compute covariance, got {n}")

    if not assume_centered:
        X_mat = X_mat - np.mean(X_mat, axis=0)

    S = (X_mat.T @ X_mat) / float(n)
    # Guarantee exact symmetry
    return 0.5 * (S + S.T)


def is_symmetric(A: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if a square matrix is numerically symmetric.

    Parameters
    ----------
    A : np.ndarray of shape (p, p)
    tol : float, default=1e-8

    Returns
    -------
    symmetric : bool
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    max_diff = np.max(np.abs(A - A.T))
    return bool(max_diff < tol)


def make_symmetric(A: np.ndarray) -> np.ndarray:
    """
    Project matrix onto symmetric subspace: 0.5 * (A + A.T).
    """
    return 0.5 * (A + A.T)


def min_eigenvalue(A: np.ndarray) -> float:
    """
    Compute the minimum eigenvalue of a symmetric matrix.
    """
    A_sym = make_symmetric(A)
    eigs = np.linalg.eigvalsh(A_sym)
    return float(np.min(eigs))


def is_pd(A: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if matrix is strictly positive definite (Theta > 0).

    Verifies both numerical symmetry and that min(eig(A)) > tol.
    """
    if not is_symmetric(A, tol=1e-6):
        return False
    return min_eigenvalue(A) > tol


def matrix_logdet(A: np.ndarray) -> float:
    """
    Compute log(det(A)) using np.linalg.slogdet for numerical stability.

    Raises ValueError if A is not positive definite (sign <= 0).
    """
    A_sym = make_symmetric(A)
    sign, logdet = np.linalg.slogdet(A_sym)
    if sign <= 0:
        raise ValueError(f"Matrix is not positive definite: det sign is {sign}, logdet={logdet}")
    return float(logdet)


def count_nonzero_edges(Theta: np.ndarray, tol: float = 1e-5) -> int:
    """
    Count the number of non-zero off-diagonal edges in precision matrix Theta.

    An edge exists between gene i and gene j (i != j) if |Theta_ij| > tol.
    Counts undirected edges: each pair (i, j) with i < j counted once.

    Parameters
    ----------
    Theta : np.ndarray of shape (p, p)
    tol : float, default=1e-5

    Returns
    -------
    n_edges : int
    """
    p = Theta.shape[0]
    upper_mask = np.triu(np.ones((p, p), dtype=bool), k=1)
    return int(np.sum(np.abs(Theta[upper_mask]) > tol))


def edge_sparsity(Theta: np.ndarray, tol: float = 1e-5) -> float:
    """
    Compute the off-diagonal sparsity fraction: (zero off-diag edges) / (total off-diag edges).
    Returns value between 0.0 (fully dense) and 1.0 (empty graph).
    """
    p = Theta.shape[0]
    total_edges = p * (p - 1) // 2
    if total_edges == 0:
        return 1.0
    active_edges = count_nonzero_edges(Theta, tol=tol)
    return float(1.0 - (active_edges / total_edges))


def compute_graph_stats(Theta: np.ndarray, tol: float = 1e-5) -> dict:
    """
    Compute degree and topology summary statistics of the precision graph.
    """
    p = Theta.shape[0]
    # Binary adjacency without diagonal
    adj = (np.abs(Theta) > tol).astype(int)
    np.fill_diagonal(adj, 0)

    degrees = np.sum(adj, axis=1)
    total_possible = p * (p - 1) // 2
    active_edges = int(np.sum(degrees) // 2)

    return {
        "n_nodes": int(p),
        "total_possible_edges": int(total_possible),
        "n_nonzero_edges": int(active_edges),
        "sparsity_ratio": float(1.0 - active_edges / max(1, total_possible)),
        "min_degree": int(np.min(degrees)),
        "max_degree": int(np.max(degrees)),
        "mean_degree": float(np.mean(degrees)),
        "isolated_nodes": int(np.sum(degrees == 0)),
    }
