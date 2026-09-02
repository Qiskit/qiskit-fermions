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

from functools import cache

from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit_fermions.mappers import map_edge_vertex_generators
from qiskit_fermions.mappers.library import edge_vertex_jordan_wigner
from qiskit_fermions.operators import EdgeAction, EdgeVertexOperator


def jordan_wigner_nearest_neighbor(op: EdgeVertexOperator, num_qubits: int) -> SparsePauliOp:
    """Custom Jordan-Wigner transformation for nearest neighbor interactions.

    This is a deliberately minimal implementation, written to exercise
    :func:`~qiskit_fermions.mappers.map_edge_vertex_generators`. It handles only nearest-neighbor
    interactions and is single-threaded, with none of the memory bounding of the real thing. Use
    :func:`~qiskit_fermions.mappers.library.edge_vertex_jordan_wigner` instead of copying this.
    """

    @cache
    def map_action(mode: EdgeAction) -> SparsePauliOp:
        left, right = mode
        if left == right:
            return SparsePauliOp.from_sparse_list([("Z", [left], 1.0)], num_qubits=num_qubits)
        if abs(left - right) != 1:
            raise NotImplementedError("This mapping only handles nearest neighbor interactions")

        # The index order must be compared, not just differenced: an edge operator is antisymmetric
        # (`E_lr = -E_rl`), so the sign depends on the orientation while the Pauli string does not.
        # Branching on `abs(left - right)` alone cannot see that, and so cannot be correct for both
        # orientations.
        lo, hi = min(left, right), max(left, right)
        coeff = -1.0 if left < right else 1.0
        return SparsePauliOp.from_sparse_list([("YX", [lo, hi], coeff)], num_qubits=num_qubits)

    return map_edge_vertex_generators(
        op,
        map_action,
        lambda: SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits=num_qubits),
    )


def test_jordan_wigner():
    op = EdgeVertexOperator.from_dict(
        {
            ((0, 0),): 2.0,
            ((0, 1),): 0.5,
            ((1, 1), (1, 2)): 1.0,
        }
    )
    num_qubits = 4
    qop = jordan_wigner_nearest_neighbor(op, num_qubits)
    assert isinstance(qop, SparsePauliOp)
    expected = SparsePauliOp.from_sparse_list(
        [("Z", [0], 2), ("YX", [0, 1], -0.5), ("XX", [1, 2], 1j)],
        num_qubits,
    )
    diff = (qop - expected).simplify()
    assert diff == SparsePauliOp.from_sparse_list([], num_qubits)


def test_jordan_wigner_matches_library_implementation():
    # The helper above is minimal, but within the nearest-neighbor cases it does handle it must agree
    # with the real mapper -- including for the reversed index order, which the earlier version of
    # this helper got wrong because it branched on `abs(left - right)`.
    op = EdgeVertexOperator.from_dict(
        {
            ((0, 0),): 2.0,
            ((0, 1),): 0.5,
            ((1, 0),): -1.0j,
            ((1, 1), (1, 2)): 1.0,
        }
    )
    num_qubits = 4
    custom = SparseObservable.from_sparse_pauli_op(jordan_wigner_nearest_neighbor(op, num_qubits))
    assert (custom - edge_vertex_jordan_wigner(op, num_qubits)).simplify() == SparseObservable.zero(
        num_qubits
    )
