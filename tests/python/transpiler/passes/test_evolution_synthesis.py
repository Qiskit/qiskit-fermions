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

"""Evolution synthesis tests."""

from __future__ import annotations

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.passmanager import MultiStagePassManager
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    MapperFnEvolutionSynthesis,
    TrivialF2QLayout,
)


def test_evolution_gate_synthesis():
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

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 16


def test_custom_qubit_ordering():
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

    custom_qubit_ordering = [0, 2, 1, 3]

    def custom_qubit_ordering_mapper_fn(op, num_qubits):
        relabeled = op.relabel_modes(custom_qubit_ordering)
        return jordan_wigner(relabeled, num_qubits)

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(custom_qubit_ordering_mapper_fn)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    qu_circ = pm.run(circ)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 4


def _single_evolution_circuit(time: float = 1.5) -> FermionicCircuit:
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=time), circ.modes)
    return circ


def _run_with_product_formula(product_formula) -> object:
    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner, product_formula)
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    return pm.run(_single_evolution_circuit())


def test_default_product_formula_is_none():
    """Omitting ``product_formula`` leaves the emitted gate on its default synthesis."""
    plugin = MapperFnEvolutionSynthesis(jordan_wigner)
    assert plugin.product_formula is None

    qu_circ = _run_with_product_formula(None)
    (instruction,) = qu_circ.data
    assert isinstance(instruction.operation, PauliEvolutionGate)
    # a PauliEvolutionGate with no explicit synthesis falls back to a single-rep LieTrotter
    assert isinstance(instruction.operation.synthesis, LieTrotter)
    assert instruction.operation.synthesis.reps == 1


def test_product_formula_is_forwarded_to_pauli_evolution_gate():
    """An explicit product formula is attached to the emitted PauliEvolutionGate verbatim."""
    product_formula = SuzukiTrotter(order=2, reps=3)
    qu_circ = _run_with_product_formula(product_formula)
    (instruction,) = qu_circ.data
    assert isinstance(instruction.operation, PauliEvolutionGate)
    assert instruction.operation.synthesis is product_formula


def test_higher_order_product_formula_deepens_synthesis():
    """A higher-order / repeated product formula yields a deeper decomposition."""
    default_circ = _run_with_product_formula(None).decompose(reps=2)
    suzuki_circ = _run_with_product_formula(SuzukiTrotter(order=2, reps=3)).decompose(reps=2)

    two_qubit = lambda instr: len(instr.qubits) == 2  # noqa: E731
    assert suzuki_circ.depth(two_qubit) > default_circ.depth(two_qubit)


def test_product_formula_via_config_kwargs():
    """The product formula flows through the ``F2QSynthesisConfig`` keyword-argument form."""
    product_formula = SuzukiTrotter(order=2, reps=2)
    config = {"Evolution": ("MapperFn", (jordan_wigner,), {"product_formula": product_formula})}
    synth = F2QSynthesis(config)
    assert synth.methods["Evolution"].product_formula is product_formula

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    qu_circ = pm.run(_single_evolution_circuit())
    (instruction,) = qu_circ.data
    assert instruction.operation.synthesis is product_formula
