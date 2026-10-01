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

"""FermionicCircuit tests."""

from __future__ import annotations

import pickle

import pytest
from qiskit.circuit import (
    Barrier,
    CircuitInstruction,
    ClassicalRegister,
    Measure,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import XGate
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    FermionicMeasure,
    InitializeModes,
    OrbitalRotation,
)
from qiskit_fermions.operators import FermionOperator, ann, cre
from qiskit_fermions.transpiler import FermionicCircuitToDAG

from ..utils import random_unitary


def test_invalid_gate():
    circ = FermionicCircuit(1)
    with pytest.raises(ValueError):
        circ.append(XGate(), circ.modes)


def _evolution_circuit() -> FermionicCircuit:
    """Builds a small circuit holding an ``Evolution`` gate over a ``FermionOperator``."""
    num_modes = 4
    hamil = FermionOperator.from_dict(
        {(cre(0), ann(2)): 2.0, (cre(2), ann(0)): 2.0, (cre(1), ann(3)): -2.0}
    )
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=1.5), circ.modes)
    circ.barrier()
    return circ


def _barrier_nodes(circ: FermionicCircuit) -> list[CircuitInstruction]:
    """Returns every barrier instruction held by ``circ``."""
    return [instr for instr in circ._inner.data if isinstance(instr.operation, Barrier)]


def test_barrier_all_modes():
    """A barrier without arguments spans every mode of the circuit."""
    circ = FermionicCircuit(4)
    circ.barrier()

    assert circ.count_ops() == {"barrier": 1}
    (barrier,) = _barrier_nodes(circ)
    assert list(barrier.qubits) == circ.modes


def test_barrier_subset_of_modes():
    """A barrier can be restricted to a subset of the modes."""
    circ = FermionicCircuit(4)
    circ.barrier(circ.register[1], circ.register[2])

    (barrier,) = _barrier_nodes(circ)
    assert list(barrier.qubits) == [circ.register[1], circ.register[2]]
    assert barrier.operation.num_qubits == 2


def test_barrier_label():
    """A barrier carries the label it was given."""
    circ = FermionicCircuit(2)
    circ.barrier(label="sync")

    (barrier,) = _barrier_nodes(circ)
    assert barrier.operation.label == "sync"


def test_barrier_survives_decompose():
    """Decomposing a circuit leaves its barriers in place."""
    circ = _evolution_circuit()
    decomposed = circ.decompose()

    assert circ.count_ops()["barrier"] == 1
    assert decomposed.count_ops()["barrier"] == 1


def test_pickle():
    """Regression test for https://github.com/Qiskit/qiskit-fermions/issues/225.

    A ``FermionicCircuit`` holding an ``Evolution`` gate failed to pickle because the native
    operator it wraps carried no pickle protocol, and separately because its ``__module__``
    was not resolvable via ``importlib`` (see :mod:`qiskit_fermions`'s ``sys.modules`` aliasing).
    """
    circ = _evolution_circuit()
    reconstructed = pickle.loads(pickle.dumps(circ))

    assert reconstructed.modes == circ.modes
    assert reconstructed.count_ops() == circ.count_ops() == {"Evolution": 1, "barrier": 1}


def test_pickle_dag():
    """Same as :func:`test_pickle`, but for the ``FermionicDAGCircuit`` conversion."""
    dag = FermionicCircuitToDAG().run(_evolution_circuit())
    reconstructed = pickle.loads(pickle.dumps(dag))

    original_nodes = dag.named_nodes("Evolution")
    reconstructed_nodes = reconstructed.named_nodes("Evolution")
    assert len(original_nodes) == len(reconstructed_nodes) == 1
    # the barrier the circuit also carries has to survive the round trip alongside the gate
    assert len(dag.op_nodes()) == len(reconstructed.op_nodes()) == 2

    original_op = original_nodes[0].op.operator
    reconstructed_op = reconstructed_nodes[0].op.operator
    assert original_op.equiv(reconstructed_op)


def _two_gate_circuit() -> FermionicCircuit:
    """Builds a 4-mode circuit with two distinguishable gates on *different* mode subsets.

    The disjoint placements make both the instruction order and the mode mapping observable after a
    ``compose`` or ``repeat``, which a single full-width gate could not.
    """
    hamil = FermionOperator.from_dict({(cre(0), ann(1)): 1.0, (cre(1), ann(0)): 1.0})
    circ = FermionicCircuit(4)
    circ.append(Evolution(2, hamil, time=0.25), [circ.modes[0], circ.modes[1]])
    circ.append(Evolution(2, hamil, time=0.75), [circ.modes[2], circ.modes[3]])
    return circ


def _times_and_modes(circuit: FermionicCircuit) -> list[tuple[float, list[int]]]:
    """Returns each gate's evolution time and absolute mode indices, in circuit order.

    Barriers are skipped: they carry no evolution time, so they have no entry here. Use
    ``count_ops`` when the barriers themselves are what a test is about.
    """
    return [
        (
            instruction.operation.params[0],
            [circuit._inner.find_bit(mode).index for mode in instruction.qubits],
        )
        for instruction in circuit._inner.data
        if not isinstance(instruction.operation, Barrier)
    ]


def test_compose():
    circ = _evolution_circuit()
    other = _evolution_circuit()

    composed = circ.compose(other)

    assert composed.count_ops() == {"Evolution": 2, "barrier": 2}
    # the operands are left untouched by the out-of-place form
    assert circ.count_ops() == other.count_ops() == {"Evolution": 1, "barrier": 1}
    # `QuantumCircuit.compose` hands back a circuit with a fresh register, so the wrapper has to
    # re-sync it; otherwise the returned circuit's modes would not be this circuit's modes
    assert composed.register is circ.register
    assert composed.modes == circ.modes


def test_compose_front():
    circ = _two_gate_circuit()
    other = _evolution_circuit()

    assert _times_and_modes(circ.compose(other, front=True))[0] == (1.5, [0, 1, 2, 3])
    assert _times_and_modes(circ.compose(other))[-1] == (1.5, [0, 1, 2, 3])


def test_compose_inplace():
    circ = _evolution_circuit()

    assert circ.compose(_evolution_circuit(), inplace=True) is None
    assert circ.count_ops() == {"Evolution": 2, "barrier": 2}


def test_compose_fargs():
    circ = FermionicCircuit(4)
    hamil = FermionOperator.from_dict({(cre(0), ann(1)): 1.0, (cre(1), ann(0)): 1.0})
    narrow = FermionicCircuit(2)
    narrow.append(Evolution(2, hamil, time=0.5), narrow.modes)

    composed = circ.compose(narrow, [circ.modes[1], circ.modes[3]])

    assert _times_and_modes(composed) == [(0.5, [1, 3])]


def test_compose_invalid_type():
    circ = _evolution_circuit()

    # a raw `QuantumCircuit` would let `QuantumCircuit.compose` inline qubit-based instructions,
    # bypassing the `FermionicGate` guard that `append` enforces
    with pytest.raises(ValueError):
        circ.compose(QuantumCircuit(4))


def test_compose_too_wide():
    circ = _evolution_circuit()

    with pytest.raises(CircuitError):
        circ.compose(FermionicCircuit(8))


def test_repeat():
    circ = _evolution_circuit()

    repeated = circ.repeat(3)

    assert repeated.count_ops() == {"Evolution": 3, "barrier": 3}
    assert circ.count_ops() == {"Evolution": 1, "barrier": 1}
    assert repeated.register is circ.register
    assert repeated.modes == circ.modes


def test_repeat_once():
    circ = _evolution_circuit()

    repeated = circ.repeat(1)

    assert repeated is not circ
    assert repeated.count_ops() == circ.count_ops() == {"Evolution": 1, "barrier": 1}
    assert _times_and_modes(repeated) == _times_and_modes(circ)


def test_repeat_zero():
    """Repeating zero times yields an empty circuit, which is the identity."""
    circ = _evolution_circuit()

    repeated = circ.repeat(0)

    assert repeated.count_ops() == {}
    assert repeated.modes == circ.modes


def test_repeat_negative():
    """A negative count is a caller error, rather than silently yielding the identity."""
    circ = _evolution_circuit()

    with pytest.raises(ValueError):
        circ.repeat(-1)


def test_repeat_empty_circuit():
    circ = FermionicCircuit(4)

    repeated = circ.repeat(3)

    assert repeated.count_ops() == {}
    assert len(repeated.register) == 4


def test_repeat_preserves_order_and_modes():
    """Each repetition must replay the instructions in order, on their original modes."""
    circ = _two_gate_circuit()

    repeated = circ.repeat(3)

    assert _times_and_modes(repeated) == [(0.25, [0, 1]), (0.75, [2, 3])] * 3


def test_repeat_preserves_metadata():
    circ = _evolution_circuit()
    circ.metadata = {"answer": 42}

    assert circ.repeat(2).metadata == {"answer": 42}


def test_repeat_insert_barriers():
    """Barriers go between repetitions, never after the last one."""
    circ = _two_gate_circuit()

    repeated = circ.repeat(3, insert_barriers=True)

    assert repeated.count_ops()["barrier"] == 2
    assert not isinstance(repeated._inner.data[-1].operation, Barrier)
    # the gates themselves are untouched by the barriers
    assert _times_and_modes(repeated) == [(0.25, [0, 1]), (0.75, [2, 3])] * 3


def test_repeat_insert_barriers_single_repetition():
    """A single repetition has no interior boundary, so it gets no barrier."""
    circ = _two_gate_circuit()

    assert "barrier" not in circ.repeat(1, insert_barriers=True).count_ops()


def test_repeat_transpiles_through_the_preset_pipeline():
    """A repeated circuit reaches the fermion-to-qubit stage without extra configuration.

    ``F2QSynthesis`` dispatches on gate name against a fixed plugin set and does not recurse into
    definitions, so a per-repetition wrapper gate would need its own registered plugin. Flattening
    keeps the repeated circuit made of gates the preset pipeline already knows.
    """
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    # An `OrbitalRotation` is lowered by the Givens decomposition, whose emitted basis gates do not
    # depend on how Qiskit synthesizes a `PauliEvolutionGate`. An `Evolution` would make this test
    # track that synthesis choice instead of the scaling it is about.
    circ = FermionicCircuit(4)
    circ.append(OrbitalRotation(random_unitary(2, seed=3)), [circ.modes[0], circ.modes[1]])

    # the barriers keep `MergeOrbitalRotations` from fusing the repetitions back into one rotation,
    # which is what leaves the scaling observable at the qubit level
    single = generate_preset_jw_pass_manager().run(circ.repeat(1, insert_barriers=True)).count_ops()
    doubled = (
        generate_preset_jw_pass_manager().run(circ.repeat(2, insert_barriers=True)).count_ops()
    )

    # compare the gates the rotation lowers to, without naming them: a barrier is bookkeeping rather
    # than a lowered gate, and only the repeated circuit carries one
    single.pop("barrier", None)
    doubled.pop("barrier", None)

    assert set(single) == set(doubled)
    for gate, count in single.items():
        assert doubled[gate] == 2 * count


# --- measurements ---------------------------------------------------------------------------------


def _measured_modes(circ: FermionicCircuit) -> list[tuple[int, int]]:
    """Returns the ``(mode index, clbit index)`` pair of every measurement, in circuit order."""
    return [
        (circ._inner.find_bit(instr.qubits[0]).index, circ._inner.find_bit(instr.clbits[0]).index)
        for instr in circ._inner.data
        if isinstance(instr.operation, FermionicMeasure)
    ]


def test_measure_into_explicit_register():
    """``measure`` writes the chosen modes into the chosen classical bits."""
    circ = FermionicCircuit(4)
    creg = ClassicalRegister(2, "c")
    circ.add_register(creg)
    circ.measure([0, 2], creg)

    assert circ.count_ops() == {"measure": 2}
    assert circ._inner.num_clbits == 2
    assert circ._inner.cregs == [creg]
    # the pairing follows the argument order, not the mode order
    assert _measured_modes(circ) == [(0, 0), (2, 1)]


def test_measure_appends_a_fermionic_instruction():
    """The appended op must be this package's instruction, not Qiskit's qubit-level ``Measure``.

    ``F2QSynthesis`` dispatches on the operation's class name, so a qubit ``Measure`` would either be
    rejected as unsupported or silently miss its plugin. It is also a singleton whose ``type()`` name
    is ``_SingletonMeasure``, which no entry point could be keyed on.
    """
    circ = FermionicCircuit(1)
    creg = ClassicalRegister(1, "c")
    circ.add_register(creg)
    circ.measure(0, 0)

    (instruction,) = circ._inner.data
    assert isinstance(instruction.operation, FermionicMeasure)
    # `FermionicMeasure` subclasses Qiskit's singleton `Measure`, so `type(...)` is a generated
    # `_Singleton...` wrapper; `base_class` is the user-facing class and is what the synthesis stage
    # keys its plugins on.
    assert instruction.operation.base_class is FermionicMeasure
    assert instruction.operation.base_class.__name__ == "FermionicMeasure"


def test_measure_all_adds_register_and_pairs_modes_to_bits():
    """``measure_all`` reads every mode out into a like-indexed bit of a fresh ``meas`` register."""
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    assert circ.measure_all() is None

    assert circ.count_ops() == {"measure": num_modes}
    assert [reg.name for reg in circ._inner.cregs] == ["meas"]
    assert circ._inner.num_clbits == num_modes
    assert _measured_modes(circ) == [(idx, idx) for idx in range(num_modes)]


def test_measure_all_inserts_no_barrier():
    """Unlike Qiskit's ``measure_all``, no barrier is inserted.

    Pinned because it is a deliberate divergence: a barrier is one ``circuit.barrier()`` call away, so
    ``measure_all`` stays purely additive rather than bundling an optimization boundary into a readout.
    """
    circ = FermionicCircuit(2)
    circ.measure_all()

    assert "barrier" not in circ.count_ops()


def test_measure_all_not_inplace_leaves_the_original_alone():
    """``inplace=False`` returns a measured copy without touching the receiver."""
    circ = FermionicCircuit(2)
    circ.append(InitializeModes([1, 0]), circ.modes)

    measured = circ.measure_all(inplace=False)

    assert measured is not None
    assert measured is not circ
    assert measured.count_ops() == {"InitializeModes": 1, "measure": 2}
    # the receiver gained neither the measurements nor the register
    assert circ.count_ops() == {"InitializeModes": 1}
    assert circ._inner.cregs == []
    assert circ._inner.num_clbits == 0
    # the copy keeps acting on this circuit's modes
    assert measured.modes == circ.modes


def test_measure_all_without_add_bits_uses_existing_clbits():
    """``add_bits=False`` measures into the bits already on the circuit, adding no register."""
    num_modes = 3
    circ = FermionicCircuit(num_modes)
    creg = ClassicalRegister(num_modes, "c")
    circ.add_register(creg)

    circ.measure_all(add_bits=False)

    assert [reg.name for reg in circ._inner.cregs] == ["c"]
    assert _measured_modes(circ) == [(idx, idx) for idx in range(num_modes)]


def test_measure_all_without_add_bits_needs_enough_clbits():
    """Measuring every mode into too few existing bits is rejected rather than truncated."""
    circ = FermionicCircuit(4)
    circ.add_register(ClassicalRegister(2, "c"))

    with pytest.raises(ValueError, match="at least as many classical bits"):
        circ.measure_all(add_bits=False)


def test_measure_all_twice_is_rejected():
    """A second ``measure_all()`` collides with the ``meas`` register it created the first time.

    Qiskit's own duplicate-register guard is what reports this, which is both accurate and specific, so
    no bespoke check is layered on top. ``add_bits=False`` remains the way to read out twice.
    """
    circ = FermionicCircuit(2)
    circ.measure_all()

    with pytest.raises(CircuitError, match="already exists"):
        circ.measure_all()

    # the documented escape hatch: measure again into the bits that already exist
    circ.measure_all(add_bits=False)
    assert circ.count_ops() == {"measure": 4}


def test_partial_measure_does_not_block_measure_all():
    """Measuring one mode early must not prevent a later full readout.

    A guard keyed on "this circuit already has measurements" would have broken this, which is why the
    duplicate-register error is relied on instead.
    """
    circ = FermionicCircuit(2)
    creg = ClassicalRegister(1, "c")
    circ.add_register(creg)
    circ.measure(0, creg[0])

    circ.measure_all()

    assert circ.count_ops() == {"measure": 3}
    assert [reg.name for reg in circ._inner.cregs] == ["c", "meas"]


def test_measure_mismatched_widths_is_rejected():
    """Measuring n modes into a different number of bits is an error, not a silent truncation."""
    circ = FermionicCircuit(3)
    circ.add_register(ClassicalRegister(2, "c"))

    with pytest.raises(CircuitError):
        circ.measure([0, 1, 2], [0, 1])


def test_add_register_rejects_a_quantum_register():
    """Only classical registers can be added; the mode register is fixed at construction."""
    circ = FermionicCircuit(2)

    with pytest.raises(ValueError, match="Unsupported register type"):
        circ.add_register(QuantumRegister(2, "q"))


def test_append_stays_restricted_to_unitary_gates():
    """``append`` takes only fermionic gates, so measurements go through ``measure``.

    Keeping the guard narrow means ``append`` remains the "apply a unitary" entry point, with a single
    sanctioned route for the one non-unitary instruction. A qubit gate is refused as it always was.
    """
    circ = FermionicCircuit(2)
    creg = ClassicalRegister(1, "c")
    circ.add_register(creg)

    with pytest.raises(ValueError, match="Unsupported instruction type"):
        circ.append(FermionicMeasure(), [circ.modes[1]], [creg[0]])

    with pytest.raises(ValueError, match="Unsupported instruction type"):
        circ.append(XGate(), circ.modes[:1])

    # the supported route
    circ.measure(1, creg[0])
    assert _measured_modes(circ) == [(1, 0)]


def test_measure_renders_as_a_measurement():
    """The drawers must render a mode measurement with the usual measurement symbol.

    Qiskit's drawers dispatch on ``isinstance(op, Measure)``, so a measurement that did not subclass it
    would be drawn as a generic boxed instruction with no classical-wire drop. The text drawer is the
    cheapest way to pin that: ``M`` plus a clbit column only appear on the measurement path.
    """
    circ = FermionicCircuit(2)
    circ.measure_all()

    assert isinstance(FermionicMeasure(), Measure)

    drawing = str(circ.draw("text", fold=-1))
    # the measurement box, and the classical wire its outcome drops onto, are both drawn only on the
    # measurement path -- a generic instruction would be a labelled box with no clbit connection
    assert "┤M├" in drawing
    assert "meas: 2/" in drawing
    assert "╩" in drawing


def test_measure_survives_decompose_and_pickle():
    """A measured circuit round-trips through ``decompose`` and ``pickle`` with its bits intact."""
    circ = FermionicCircuit(2)
    circ.append(InitializeModes([1, 0]), circ.modes)
    circ.measure_all()

    assert _measured_modes(circ.decompose()) == [(0, 0), (1, 1)]

    reconstructed = pickle.loads(pickle.dumps(circ))
    assert _measured_modes(reconstructed) == [(0, 0), (1, 1)]
    assert reconstructed._inner.num_clbits == 2


def test_measure_survives_dag_roundtrip():
    """The classical bits survive the conversion to a DAG, which is what the transpiler consumes."""
    circ = FermionicCircuit(2)
    circ.measure_all()

    dag = FermionicCircuitToDAG().run(circ)

    nodes = list(dag.topological_op_nodes())
    assert [node.op.base_class for node in nodes] == [FermionicMeasure] * 2
    assert all(len(node.cargs) == 1 for node in nodes)
    assert [
        (dag.find_bit(node.qargs[0]).index, dag.find_bit(node.cargs[0]).index) for node in nodes
    ] == [(0, 0), (1, 1)]


def test_repeat_of_a_measured_circuit_reuses_its_clbits():
    """Repetition re-measures into the same bits, so only the last readout is observable.

    ``repeat`` flattens, and a classical register cannot hold one readout per repetition. Pinned so the
    behaviour is a documented consequence rather than a surprise; measure *after* repeating to get a
    single well-defined readout.
    """
    circ = FermionicCircuit(2)
    circ.measure_all()

    repeated = circ.repeat(2)

    assert repeated.count_ops() == {"measure": 4}
    assert repeated._inner.num_clbits == 2
    assert _measured_modes(repeated) == [(0, 0), (1, 1), (0, 0), (1, 1)]


def test_compose_of_measured_circuits_maps_onto_the_target_register():
    """Composition maps the operand's measurements onto the target's classical bits."""
    left = FermionicCircuit(2)
    left.add_register(ClassicalRegister(2, "meas"))
    left.measure_all(add_bits=False)

    right = FermionicCircuit(2)
    right.add_register(ClassicalRegister(2, "other"))
    right.measure_all(add_bits=False)

    composed = left.compose(right)

    assert composed.count_ops() == {"measure": 4}
    # only the target's register survives; the operand's bits are mapped onto it positionally
    assert [reg.name for reg in composed._inner.cregs] == ["meas"]
    assert _measured_modes(composed) == [(0, 0), (1, 1), (0, 0), (1, 1)]
