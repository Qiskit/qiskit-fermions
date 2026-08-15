# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Shared matrix-exponential helpers for fermionic circuit simulation."""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse.linalg


def _expm_multiply_with_trace(
    operator: scipy.sparse.linalg.LinearOperator, vector: np.ndarray, trace: complex
) -> np.ndarray:
    """Apply a matrix exponential using the exact trace of the compiled FCI operator."""
    return cast(
        np.ndarray,
        scipy.sparse.linalg.expm_multiply(operator, vector, traceA=trace),
    )
