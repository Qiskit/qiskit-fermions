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

"""Utility methods for transpiler passes."""

from __future__ import annotations

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGOpNode

from .. import F2QLayout


def _parse_node_indices(
    in_node: DAGOpNode, f2q_layout: F2QLayout
) -> tuple[set[QuantumRegister], list[int]]:
    """Parse the mode indices that a :ext:class:`~qiskit.dagcircuit.DAGOpNode` acts upon.

    Given a node of a :ext:class:`~qiskit.dagcircuit.DAGCircuit` and a :type:`.F2QLayout` this
    function parses the node's indices and maps them to the global mode indices of the
    respective register from the ``f2q_layout``.

    Args:
        in_node: the node whose indices to parse.
        f2q_layout: the mapping of fermionic to qubit registers.

    Returns:
        The set of fermionic mode registers and global mode indices acted upon by ``in_node``.
    """
    encountered_mode_registers: set[QuantumRegister] = set()
    global_mode_indices = []
    for fermion in in_node.qargs:
        for mreg in f2q_layout:
            if fermion in mreg:
                encountered_mode_registers.add(mreg)
                global_mode_indices.append(mreg.index(fermion))
                break

    return encountered_mode_registers, global_mode_indices
