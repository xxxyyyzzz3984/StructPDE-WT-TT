from __future__ import annotations

import numpy as np
from scipy import sparse


def diffuse(source: np.ndarray, laplacian: sparse.csr_matrix, steps: int = 5, dt: float = 0.15, decay: float = 0.08) -> np.ndarray:
    field = source.astype(float).copy()
    for _ in range(steps):
        field = field - dt * laplacian.dot(field) - dt * decay * field + dt * source
    return np.maximum(field, 0.0)


def receptor_activation(ligand_field: np.ndarray, receptor_expr: np.ndarray, structure_score: float, beta: float = 2.0) -> np.ndarray:
    signal = 1.0 / (1.0 + np.exp(-beta * structure_score * ligand_field))
    return receptor_expr * signal

