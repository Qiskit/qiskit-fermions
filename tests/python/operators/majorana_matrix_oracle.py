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

"""A dense-matrix oracle for edge- and transfer-vertex operators.

Both operator families are *defined* as products of Majorana operators, so building explicit
matrices for the Majoranas and multiplying them out gives a ground truth that is independent of
the operator implementations under test. This is deliberately naive: it never calls
:meth:`normal_ordered`, :meth:`simplify` or any of the algebra being verified, so a sign error in
those cannot hide here.

The Majoranas are represented in the Jordan-Wigner basis on ``num_modes`` qubits,

.. math::

    \\gamma_{2j-1} = Z \\cdots Z X_j \\, , \\qquad \\gamma_{2j} = Z \\cdots Z Y_j \\, ,

with a 1-based mode index :math:`j`.
"""

from __future__ import annotations

import numpy as np

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(mats) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for mat in mats:
        out = np.kron(out, mat)
    return out


def majorana(index: int, num_modes: int) -> np.ndarray:
    """Returns the matrix of the Majorana operator ``gamma_index`` (1-based, in ``1..2*num_modes``)."""
    mode = (index + 1) // 2  # 1-based mode index
    local = _X if index % 2 == 1 else _Y
    return _kron(
        local if qubit == mode else (_Z if qubit < mode else _I2)
        for qubit in range(1, num_modes + 1)
    )


def vertex_matrix(mode: int, num_modes: int) -> np.ndarray:
    """Returns the matrix of ``V_mode = -i gamma_{2j-1} gamma_{2j}`` for a 0-based ``mode``."""
    one_based = mode + 1
    return -1j * majorana(2 * one_based - 1, num_modes) @ majorana(2 * one_based, num_modes)


def edge_matrix(left: int, right: int, num_modes: int) -> np.ndarray:
    """Returns the matrix of ``E_{left,right} = -i gamma_{2j-1} gamma_{2k-1}`` for 0-based modes."""
    if left == right:
        return vertex_matrix(left, num_modes)
    return -1j * majorana(2 * (left + 1) - 1, num_modes) @ majorana(2 * (right + 1) - 1, num_modes)


def transfer_matrix(left: int, right: int, num_modes: int) -> np.ndarray:
    """Returns the matrix of ``T_{left,right} = (i/2) gamma_{2j} gamma_{2k-1}`` for 0-based modes."""
    if left == right:
        return vertex_matrix(left, num_modes)
    return 0.5j * majorana(2 * (left + 1), num_modes) @ majorana(2 * (right + 1) - 1, num_modes)


def operator_matrix(operator, num_modes: int, action_matrix) -> np.ndarray:
    """Returns the dense matrix of ``operator``, mapping each action via ``action_matrix``."""
    dim = 2**num_modes
    total = np.zeros((dim, dim), dtype=complex)
    for actions, coeff in operator.iter_terms():
        term = np.eye(dim, dtype=complex)
        for left, right in actions:
            term = term @ action_matrix(left, right, num_modes)
        total += complex(coeff) * term
    return total
