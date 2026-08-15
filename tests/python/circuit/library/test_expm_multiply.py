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

"""Tests for matrix-exponential helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import scipy.sparse.linalg
from qiskit_fermions.circuit.library._expm_multiply import _expm_multiply_with_trace


def test_expm_multiply_uses_operator_trace():
    """The helper supplies the FCI operator trace to SciPy."""
    diagonal = np.array([2.0, 4.0])
    operator = scipy.sparse.linalg.aslinearoperator(np.diag(diagonal))
    vector = np.array([1.0, 1.0])

    with patch(
        "scipy.sparse.linalg.expm_multiply", wraps=scipy.sparse.linalg.expm_multiply
    ) as call:
        result = _expm_multiply_with_trace(operator, vector, complex(diagonal.sum()))

    assert call.call_args.kwargs["traceA"] == diagonal.sum()
    np.testing.assert_allclose(result, np.exp(diagonal))
