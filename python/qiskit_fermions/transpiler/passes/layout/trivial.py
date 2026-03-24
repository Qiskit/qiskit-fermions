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

"""A trivial fermion-to-qubit layout."""

from __future__ import annotations

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import AnalysisPass

from .. import F2QLayout


class TrivialF2QLayout(AnalysisPass):
    """TODO."""

    def run(self, dag: DAGCircuit) -> None:
        """TODO."""
        self.property_set["f2q_layout"] = F2QLayout.trivial(dag.qregs["f"])
