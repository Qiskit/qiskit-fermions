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
from qiskit.circuit import Barrier, CircuitInstruction, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import XGate
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, OrbitalRotation
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
