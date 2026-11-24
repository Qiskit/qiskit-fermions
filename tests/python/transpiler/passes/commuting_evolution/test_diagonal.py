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

"""Tests of the middle part shared by the candidates."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution._diagonal import (
    NotDiagonalizedError,
    evolve_from_clifford,
    evolve_with_pair_blocks,
)

from .groups import same_unitary
from .helpers import TIME, problem_of


class TestDiagonal:
    def test_a_clifford_that_does_not_diagonalize_raises(self):
        """An explicit exception, not an ``assert``: it must hold under ``python -O`` as well."""
        op = SparsePauliOp(["XX", "ZZ"], [0.4, 0.6])
        with pytest.raises(NotDiagonalizedError):
            evolve_from_clifford(problem_of(op), QuantumCircuit(2))

    def test_the_error_is_not_a_value_error(self):
        """A base candidate failing this check is a bug; it must not look like an expected
        failure."""
        assert not issubclass(NotDiagonalizedError, ValueError)

    @pytest.mark.parametrize("shape", ["tree", "ladder"])
    @pytest.mark.parametrize("reverse_two_body", [False, True])
    def test_z_strings_need_no_clifford(self, shape, reverse_two_body):
        """Z-strings of weight zero to three (a phase, rotations, ZZ layers, parity trees) with the
        empty Clifford."""
        op = SparsePauliOp(
            ["IIIII", "IIIIZ", "IIIZZ", "ZZIII", "IIZZZ"], [0.1, 0.3, -0.5, 0.2, 0.7]
        )
        circuit = evolve_from_clifford(
            problem_of(op), QuantumCircuit(5), shape=shape, reverse_two_body=reverse_two_body
        )
        assert same_unitary(circuit, op, TIME)

    def test_a_block_holds_an_entangled_pair(self):
        """``{XX, YY, ZZ}`` needs no Clifford at all, and the block has at most three two-qubit
        gates."""
        op = SparsePauliOp(["XX", "YY", "ZZ"], [0.4, -0.3, 0.6])
        circuit = evolve_with_pair_blocks(problem_of(op), QuantumCircuit(2))
        assert same_unitary(circuit, op, TIME)
        assert circuit.num_nonlocal_gates() <= 3

    def test_blocks_absorb_single_qubit_images_and_leave_the_rest_diagonal(self):
        """A Z-string on a block's pair and an image on one of its qubits join the block; the rest
        is outside it."""
        terms = [
            ("XX", [0, 1], 0.4),
            ("ZZ", [0, 1], 0.7),  # a Z-string on exactly the pair of a block
            ("XX", [2, 3], 0.3),
            ("X", [2], 0.2),  # an image on one qubit of a block
            ("X", [4], 0.5),  # a single-qubit image outside every block
            ("ZZ", [5, 6], 0.6),  # a Z-string outside every block
        ]
        op = SparsePauliOp.from_sparse_list(terms, num_qubits=7)
        circuit = evolve_with_pair_blocks(problem_of(op), QuantumCircuit(7))
        assert same_unitary(circuit, op, TIME)

    def test_a_wider_image_that_is_not_diagonal_raises(self):
        op = SparsePauliOp(["XXX"], [0.4])
        with pytest.raises(NotDiagonalizedError):
            evolve_with_pair_blocks(problem_of(op), QuantumCircuit(3))
