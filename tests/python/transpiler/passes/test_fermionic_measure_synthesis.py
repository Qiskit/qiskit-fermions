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

"""Tests for the fermionic measurement synthesis plugin."""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.passmanager import MultiStagePassManager
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import InitializeModes
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    CustomF2QLayout,
    F2QSynthesis,
    F2QSynthesisPluginManager,
    TrivialF2QLayout,
    TrivialOccupationFermionicMeasureSynthesis,
    TrivialOccupationInitializeModesSynthesis,
)


def _pass_manager(layout) -> MultiStagePassManager:
    """A minimal pipeline whose synthesis stage lowers initializations and measurements."""
    synth = F2QSynthesis()
    synth.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()
    synth.methods["FermionicMeasure"] = TrivialOccupationFermionicMeasureSynthesis()
    return MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=layout,
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )


def test_measure_all_lowers_to_one_qubit_measure_each():
    """Every mode measurement becomes a qubit measurement on the qubit carrying that mode.

    Compared against a hand-built circuit rather than by counting, so that both the qubit *and* the
    classical bit of every measurement are pinned -- a count-only assertion would pass even if the
    bits were permuted.
    """
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes([1, 0, 1, 0]), circ.modes)
    circ.measure_all()

    out = _pass_manager(TrivialF2QLayout()).run(circ)

    qreg, *_ = out.qregs
    expected = QuantumCircuit(qreg, *out.cregs)
    expected.x(qreg[0])
    expected.x(qreg[2])
    for idx in range(num_modes):
        expected.measure(qreg[idx], out.clbits[idx])

    assert out == expected


def test_partial_measure_keeps_its_mode_to_clbit_pairing():
    """A measurement lowers onto the qubit of its own mode and the classical bit it was given."""
    circ = FermionicCircuit(4)
    creg = ClassicalRegister(2, "c")
    circ.add_register(creg)
    circ.measure([1, 3], creg)

    out = _pass_manager(TrivialF2QLayout()).run(circ)

    assert [
        (out.find_bit(instr.qubits[0]).index, out.find_bit(instr.clbits[0]).index)
        for instr in out.data
    ] == [(1, 0), (3, 1)]


def test_mismatched_register_lengths_are_rejected():
    """A layout that changes the register width has no single qubit per mode, so it must not lower.

    This is the executable form of the ``F2QLayout`` contract: the two sides of the mapping may differ
    in size, in which case reading a mode's occupation off one qubit is not meaningful. Lowering it
    anyway would silently measure the wrong thing.
    """
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    circ.measure_all()

    layout = CustomF2QLayout({circ.register: QuantumRegister(num_modes + 2)})

    with pytest.raises(ValueError, match="single qubit"):
        _pass_manager(layout).run(circ)


def test_plugin_is_discoverable_by_its_entry_point():
    """The plugin resolves under the name the synthesis config keys on.

    ``F2QSynthesis`` looks plugins up by the operation's ``base_class`` name. That indirection is what
    this guards: ``FermionicMeasure`` subclasses Qiskit's singleton ``Measure``, so its ``type()`` is a
    generated ``_Singleton...`` wrapper that no entry point could ever match.
    """
    manager = F2QSynthesisPluginManager()

    assert "TrivialOccupation" in manager.method_names("FermionicMeasure")
    assert (
        manager.method("FermionicMeasure", "TrivialOccupation")
        is TrivialOccupationFermionicMeasureSynthesis
    )


def test_config_driven_synthesis_selects_the_plugin():
    """The plugin can also be selected through the ``config`` mapping, by short name."""
    circ = FermionicCircuit(2)
    circ.measure_all()

    synth = F2QSynthesis({"FermionicMeasure": "TrivialOccupation"})
    out = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    ).run(circ)

    assert out.count_ops() == {"measure": 2}
