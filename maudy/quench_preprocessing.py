"""Functionality for preprocessing the quenching correction groups."""

from typing import Optional

import cvxpy as cp
import pandas as pd
import numpy as np


def left_nullspace(matrix: pd.DataFrame, atol=1e-7, rtol=0.0) -> np.ndarray:
    """Compute an approximate basis for the null space (kernel) of a matrix, SVD-based.

    Parameters
    ----------
    matrix : ndarray
        The matrix should be at most 2-D.  A 1-D array with length k
        will be treated as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than ``atol`` are considered to be zero.
    rtol : float
        The relative tolerance for a zero singular value.  Singular values less
        than the relative tolerance times the largest singular value are
        considered to be zero.

    Notes
    -----
    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than ``tol`` are considered to be zero.

    Returns
    -------
    ndarray
        If ``matrix`` is an array with shape (m, k), then the returned
        nullspace will be an array with shape ``(k, n)``, where n is the
        estimated dimension of the nullspace.

    References
    ----------
    Adapted from:
    https://scipy.github.io/old-wiki/pages/Cookbook/RankNullspace.html
    and then taken from from
    https://github.com/opencobra/memote/blob/develop/src/memote/support/consistency_helpers.py#L163
    """  # noqa: D402
    mat = np.atleast_2d(matrix)
    _, sigma, vh = np.linalg.svd(mat.T)
    tol = max(atol, rtol * sigma[0])
    num_nonzero = (sigma >= tol).sum()
    return vh[num_nonzero:].conj().T


def reduce_column_members(A: np.ndarray, tolerance: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the number of non-zero elements per column in a matrix by transforming it.

    Use convex optimization to minimize the L1-norm of each transformed vector $w = A p$,
    promoting sparsity in `W`. Satisfies the constraint `p[i] == 1` to avoid the trivial solution.

    Parameters
    ----------
    A: np.ndarray
        The input matrix to be transformed, at least two-dimensional.
    tolerance: float, default=1e-7)
        A threshold below which values are considered zero.

    Returns
    -------
    W, P: tuple[np.ndarray, np.ndarray]
        - W: The transformed matrix with reduced non-zero elements per column.
        - P: The transformation matrix used to obtain `W` from `A`.
    """
    A = np.atleast_2d(A)
    n = A.shape[1]
    W_list = []
    P_list = []
    for i in range(n):
        p = cp.Variable(n)
        w = A @ p
        objective = cp.Minimize(cp.norm1(w))
        constraints = []
        constraints.append(p[i] == 1)
        prob = cp.Problem(objective, constraints)
        _ = prob.solve()
        W_list.append(w.value)
        P_list.append(p.value)

    W = np.column_stack(W_list)
    P = np.column_stack(P_list)
    W[np.abs(W) < tolerance] = 0.0
    P[np.abs(P) < tolerance] = 0.0
    return W, P


def extract_conserved_moiety_matrix(stoichiometry: np.ndarray, met_index: list[str], tolerance: float = 1e-7) -> Optional[pd.DataFrame]:
    """Extract conserved moieties from a stoichiometric matrix.
    
    Computes the left nullspace (conserved moieities), which is then rearranged (through
    iterative convex optimization) to enforce a minimum number of elements per moeity.
    """
    left_ns = left_nullspace(stoichiometry)
    if left_ns.size == 0:
        return None
    left_ns[abs(left_ns) < tolerance] = 0
    reduced_left_ns, _ = reduce_column_members(left_ns, tolerance)
    return pd.DataFrame(reduced_left_ns, index=met_index)
