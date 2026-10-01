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

"""Fermion-to-qubit synthesis tests."""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister
from qiskit.circuit import Barrier, QuantumRegister
from qiskit.circuit.library import XGate
from qiskit.passmanager import MultiStagePassManager
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    CustomF2QLayout,
    F2QSynthesis,
    MapperFnEvolutionSynthesis,
    TrivialF2QLayout,
)


def test_missing_plugin():
    """Test the handling of a missing fermion-to-qubit plugin."""
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    time = 1.5
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=F2QSynthesis(),
        output=QuantumDAGToCircuit(),
    )

    with pytest.raises(TypeError, match="No plugin registered"):
        _ = pm.run(circ)


def test_config_tuple_with_positional_args():
    """The ``(name, args)`` config form forwards positional args to the plugin ``__init__``."""
    synth = F2QSynthesis({"Evolution": ("MapperFn", (jordan_wigner,))})
    assert isinstance(synth.methods["Evolution"], MapperFnEvolutionSynthesis)


def test_config_tuple_with_positional_and_keyword_args():
    """The ``(name, args, kwargs)`` config form forwards both positional and keyword args."""
    synth = F2QSynthesis({"Evolution": ("MapperFn", (), {"mapper_fn": jordan_wigner})})
    assert isinstance(synth.methods["Evolution"], MapperFnEvolutionSynthesis)


def test_run_rejects_non_fermionic_gate():
    """A circuit instruction that is not a FermionicGate raises during synthesis."""
    num_modes = 2
    circ = FermionicCircuit(num_modes)
    # Append a plain (non-fermionic) gate directly on the mode register.
    circ._inner.append(XGate(), [circ.register[0]])

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    with pytest.raises(ValueError, match="unsupported circuit instruction type"):
        _ = pm.run(circ)


def _barrier_circuit(num_modes: int) -> FermionicCircuit:
    """A circuit whose barrier covers only a subset of its modes."""
    hamil = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 1.0})
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(2, hamil, time=0.5), circ.modes[:2])
    circ.barrier(circ.register[0], circ.register[1])
    return circ


def _synthesize(circ: FermionicCircuit, layout) -> tuple[int, ...]:
    """Runs the synthesis stage and returns the width of every barrier in the output."""
    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=layout,
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    out = pm.run(circ)
    return tuple(
        instr.operation.num_qubits for instr in out.data if isinstance(instr.operation, Barrier)
    )


def test_run_lowers_barrier_to_mapped_register():
    """A fermionic barrier is lowered onto every qubit of the register it was mapped to.

    The barrier covers only modes 0 and 1, but the lowering deliberately widens it to the whole mapped
    register: an ``F2QLayout`` relates whole registers, so the modes it covers have no per-qubit image
    in general.
    """
    num_modes = 4
    assert _synthesize(_barrier_circuit(num_modes), TrivialF2QLayout()) == (num_modes,)


def test_run_lowers_barrier_under_non_trivial_layout():
    """The barrier spans the mapped register even when it is sized differently from the modes.

    This pins the register-wide lowering: an implementation mapping mode ``i`` onto qubit ``i`` would
    emit a width-2 barrier here (or index out of the register) instead of covering all six qubits.
    """
    num_modes, num_qubits = 4, 6
    circ = _barrier_circuit(num_modes)
    layout = CustomF2QLayout({circ.register: QuantumRegister(num_qubits)})

    assert _synthesize(circ, layout) == (num_qubits,)


def test_run_preserves_barrier_label():
    """A labelled fermionic barrier keeps its label through the synthesis stage."""
    circ = FermionicCircuit(2)
    circ.barrier(label="sync")

    synth = F2QSynthesis()
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    out = pm.run(circ)

    (barrier,) = [instr for instr in out.data if isinstance(instr.operation, Barrier)]
    assert barrier.operation.label == "sync"


def test_classical_registers_are_preserved():
    """Classical registers on the input circuit are carried over to the synthesized output."""
    hamil = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 1.0})
    num_modes = 2
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=0.5), circ.modes)
    creg = ClassicalRegister(2, "c")
    circ._inner.add_register(creg)

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)
    assert any(reg.name == "c" and len(reg) == 2 for reg in qu_circ.cregs)
