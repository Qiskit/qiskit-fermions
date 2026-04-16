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

"""A custom fermion-to-qubit layout."""

from __future__ import annotations

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import AnalysisPass

from ... import F2QLayout


class CustomF2QLayout(AnalysisPass):
    """Sets the user-provided :type:`~qiskit_fermions.transpiler.F2QLayout` in the transpiler.

    This pass simply populates the ``f2q_layout`` field of the
    :attr:`~qiskit.passmanager.PassManagerState.property_set`, with the value of :attr:`layout`.

    .. note::
       It is the user's responsibility to ensure that subsequently used transpilation passes (and
       plugins) are configured to adhere to this :type:`~qiskit_fermions.transpiler.F2QLayout`
       correctly.
    """

    def __init__(self, layout: F2QLayout) -> None:
        """Initializes the transpiler pass.

        Args:
            layout: the custom ``f2q_layout`` to use.
        """
        super().__init__()

        self.layout: F2QLayout = layout
        """The user-provided mapping of :type:`~qiskit_fermions.circuit.FermionRegister` to
        :class:`~qiskit.circuit.QuantumRegister`."""

    def run(self, dag: DAGCircuit) -> None:
        """Runs this analysis pass.

        Args:
            dag: the DAG circuit representation of a :class:`.FermionCircuit`.
        """
        self.property_set["f2q_layout"] = self.layout
