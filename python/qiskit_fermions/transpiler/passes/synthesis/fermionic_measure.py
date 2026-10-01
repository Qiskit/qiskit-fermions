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

"""Mode measurement synthesis."""

from __future__ import annotations

from qiskit.circuit import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from ... import F2QLayout
from ..utils import map_node_single_register


class TrivialOccupationFermionicMeasureSynthesis:
    """A :class:`.F2QSynthesisPlugin` for transpiling :class:`.FermionicMeasure` with trivial occupation.

    Under an occupation-basis encoding, the occupation of fermionic mode :math:`i` is the
    computational-basis value of the single qubit that mode is mapped onto. A mode measurement
    therefore lowers to exactly one qubit :external:class:`~qiskit.circuit.Measure`, writing into the
    very classical bit the :class:`.FermionicMeasure` was given.

    .. warning::
       This transpilation pass plugin makes the following assumptions:

       - an occupation-basis encoding (like Jordan-Wigner)
       - a trivial fermion-to-qubit layout (i.e. no change in their register lengths)
       - a 1-to-1 mapping of fermionic mode indices to qubit indices
    """

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction. When this plugin gets called, the
                ``in_node.op`` attribute `must` be of type :class:`.FermionicMeasure`.
            out_dag: the output qubit-based circuit.
            f2q_layout: the global transpilation :class:`~qiskit_fermions.transpiler.F2QLayout`
                setting.

        .. seealso::
           The documentation of :class:`.F2QSynthesisPlugin` for more detailed explanations of the
           arguments.

        Raises:
            NotImplementedError: when ``in_node`` acts on fermionic modes that are spread across
                multiple :type:`~qiskit_fermions.circuit.FermionicRegister` instances.
            ValueError: when the mapped qubit register does not have the same length as the fermionic
                mode register, in which case a mode has no single qubit whose measurement would yield
                its occupation.
        """
        # NOTE: ``map_node_single_register`` resolves a mode to its index in the *original* register
        # (see the ``# HACK`` in that helper). That is what makes this lowering transparent to
        # ``RelabelModes``: the measurement lands on the qubit carrying the mode it was placed on,
        # while the classical bit it writes into is carried over untouched from ``in_node``.
        freg_indices, qreg = map_node_single_register(in_node, f2q_layout)

        (freg,) = (reg for reg, mapped in f2q_layout.items() if mapped is qreg)
        if len(freg) != len(qreg):
            raise ValueError(
                f"Cannot measure a fermionic mode with a single qubit: the fermionic register "
                f"'{freg.name}' of length {len(freg)} is mapped onto a qubit register of length "
                f"{len(qreg)}. This plugin assumes an occupation-basis encoding with a 1-to-1 "
                "mapping of fermionic mode indices to qubit indices."
            )

        # a `FermionicMeasure` acts on exactly one mode and one classical bit
        out_dag.apply_operation_back(Measure(), (qreg[freg_indices[0]],), in_node.cargs)
