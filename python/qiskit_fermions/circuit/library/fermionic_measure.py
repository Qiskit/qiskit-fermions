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

"""FermionicMeasure."""

from __future__ import annotations

from qiskit.circuit import Measure

from .. import FermionicInstruction


class FermionicMeasure(FermionicInstruction, Measure):
    """A measurement of a single fermionic mode, writing its outcome into one classical bit.

    This is the fermionic counterpart of :external:class:`~qiskit.circuit.Measure`, and the one
    instruction on a :class:`.FermionicCircuit` that is not a :class:`.FermionicGate`: a measurement is
    not unitary, and :class:`~qiskit.circuit.Gate` admits no classical bits. Place it with
    :meth:`.FermionicCircuit.measure` or :meth:`.FermionicCircuit.measure_all` rather than constructing
    it directly.

    It subclasses :external:class:`~qiskit.circuit.Measure` so that the circuit drawers, which dispatch
    on ``isinstance(op, Measure)``, render it with the usual measurement symbol rather than as a
    generic boxed instruction.

    .. important::
       What the measured bit *means* is decided by the fermion-to-qubit encoding, not by this
       instruction. Under an occupation-basis encoding such as Jordan-Wigner, mode :math:`i` is carried
       by a single qubit and the bit is that mode's occupation, which is the reading
       :class:`.TrivialOccupationFermionicMeasureSynthesis` implements. A
       :type:`~qiskit_fermions.transpiler.F2QLayout` relates whole registers whose two sides may differ
       in size, so in general a mode has no single qubit to measure; an encoding for which that is the
       case needs its own synthesis plugin, and this instruction carries no encoding assumption of its
       own.

    .. seealso::
       :class:`.TrivialOccupationFermionicMeasureSynthesis` for the occupation-basis lowering, and
       :meth:`.FermionicCircuit.measure_all` for the convenient way to read out an entire circuit.
    """

    def __init__(self, label: str | None = None) -> None:
        """Initializing this instruction can be done with the arguments listed below.

        Args:
            label: the string label of this instruction.
        """
        # The name is deliberately Qiskit's own "measure": the drawers pick a gate's colour from
        # ``displaycolor[op.name]``, so this is what renders a mode measurement in the usual grey
        # rather than the generic gate colour. The name is free to be shared because the
        # fermion-to-qubit synthesis stage keys its plugins on ``base_class``, not on the name.
        super().__init__(label=label)
