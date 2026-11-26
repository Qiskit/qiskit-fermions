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

"""Tests of the packing of gates into layers."""

from __future__ import annotations

from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution._layers import pairwise_tree


def test_pairwise_tree_keeps_the_requested_qubit():
    """Five qubits: two levels of pairs, the odd one out passed on; the root is the last target or
    the first control."""
    root, gates = pairwise_tree([0, 1, 2, 3, 4], keep="target")
    assert (root, gates) == (4, [(0, 1), (2, 3), (1, 3), (3, 4)])
    root, gates = pairwise_tree([0, 1, 2, 3, 4], keep="control")
    assert (root, gates) == (0, [(0, 1), (2, 3), (0, 2), (0, 4)])
