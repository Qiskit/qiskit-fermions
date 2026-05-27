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

# ruff: noqa: D205,D212,D415
"""
=========================
Transpiler Pass Utilities
=========================

.. currentmodule:: qiskit_fermions.transpiler.passes.utils

Several convenience functions are provided to simplify the implementation of common operations
during the implementation of custom transpiler passes.

.. autosummary::
   :toctree: ../stubs/

   map_node_single_register
"""

from __future__ import annotations

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGOpNode

from .. import F2QLayout


def map_node_single_register(
    in_node: DAGOpNode, f2q_layout: F2QLayout
) -> tuple[list[int], QuantumRegister]:
    """Map the single fermionic mode register acted upon by the circuit operation.

    Given a :external:class:`~qiskit.dagcircuit.DAGOpNode` and a
    :type:`~qiskit_fermions.transpiler.F2QLayout` this function will:

    1. ensure that all acted-upon fermionic modes belong to the same register
    2. map the node's indices to their global indices with respect to said fermionic mode register
    3. map this fermionic mode register to the corresponding qubit register

    .. caution::
       The returned global fermionic mode register indices are knowingly dependent on the order of
       the fermionic modes during the construction of their original
       :type:`~qiskit_fermions.circuit.FermionicRegister`. This is unlikely to cause any issues but
       makes the usage of :class:`.RelabelModes` possibly more generally.

    Args:
        in_node: the node whose indices to parse.
        f2q_layout: the mapping of fermionic to qubit registers.

    Returns:
        The list of the global fermionic mode indices and mapped qubit register instance.

    Raises:
        NotImplementedError: when the provided node acts on fermionic modes that do not all belong
            to the same :class:`.FermionicRegister`.
    """
    encountered_fermionic_registers: set[QuantumRegister] = set()
    freg_indices = []
    for fermion in in_node.qargs:
        for freg in f2q_layout:
            if fermion in freg:
                encountered_fermionic_registers.add(freg)
                # HACK: we explicitly use the private _index attribute here to enable the general
                # usage of the RelabelModes pass. This _should_ have no adversarial effect on other
                # scenarios unless the user tampered with the FermionicRegister themselves, too.
                freg_indices.append(fermion._index)
                break

    if len(encountered_fermionic_registers) > 1:
        raise NotImplementedError(
            "Cannot map a FermionicGate acting on fermionic modes that are spread across "
            "multiple FermionicRegister instances."
        )

    freg = encountered_fermionic_registers.pop()
    qreg = f2q_layout[freg]

    return freg_indices, qreg
