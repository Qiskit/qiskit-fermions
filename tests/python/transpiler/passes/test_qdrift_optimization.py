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

"""QDrift optimization tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.circuit.library.synthesis import FermionicSuzukiTrotter
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.library import FCIDump
from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms
from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
from qiskit_fermions.transpiler.passes import QDriftTrotterization
from qiskit_fermions.transpiler.passmanager import FermionicPassManager
from scipy.linalg import expm


def test_qdrift_optimization_no_groups(subtests):
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))
    num_modes = 2 * fcidump.norb
    hamil = FermionOperator.from_fcidump(fcidump)
    hamil.groups = None
    time = 1.5
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    with subtests.test("num_terms=4"):
        num_terms = 4
        qdrift = QDriftTrotterization(num_terms)
        pm = FermionicPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    with subtests.test("num_terms=6"):
        num_terms = 6
        qdrift = QDriftTrotterization(num_terms)
        pm = FermionicPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    with subtests.test("rng seed"):
        num_terms = 2
        qdrift = QDriftTrotterization(num_terms, rng=42)
        pm = FermionicPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

        expected_gates = [
            Evolution(
                num_modes,
                FermionOperator.from_terms([(((True, 0), (True, 1), (False, 1), (False, 0)), 1.0)]),
                time=8.273087572037902,
            ),
            Evolution(
                num_modes,
                FermionOperator.from_terms([(((True, 2), (True, 0), (False, 0), (False, 2)), 1.0)]),
                time=8.273087572037902,
            ),
        ]

        for actual, expected in zip(qdrift_circ._inner.data, expected_gates, strict=True):
            assert actual.operation.operator.equiv(expected.operator)
            assert np.isclose(actual.params[0], expected.params[0])

    with subtests.test("rng seed"):
        num_terms = 2
        qdrift = QDriftTrotterization(num_terms, rng=np.random.default_rng(43))
        pm = FermionicPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

        expected_gates = [
            Evolution(
                num_modes,
                FermionOperator.from_terms([(((True, 1), (True, 0), (False, 0), (False, 1)), 1.0)]),
                time=8.273087572037902,
            ),
            Evolution(
                num_modes,
                FermionOperator.from_terms([((), 1.0)]),
                time=8.273087572037902,
            ),
        ]

        for actual, expected in zip(qdrift_circ._inner.data, expected_gates, strict=True):
            assert actual.operation.operator.equiv(expected.operator)
            assert np.isclose(actual.params[0], expected.params[0])


def test_qdrift_optimization_with_groups():
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))
    num_modes = 2 * fcidump.norb
    hamil = FermionOperator.from_fcidump(fcidump)
    normal = hamil.normal_ordered().simplify(atol=1e-16)
    group_terms_by_electronic_structure(normal, num_modes, two_body_physicist_order=False)

    time = 1.5
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, normal, time=time)
    circ.append(evo, circ.modes)

    num_terms = 5
    qdrift = QDriftTrotterization(num_terms, rng=42)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    # NOTE: the normal-ordering and subsequent simplifying of our Hamiltonian before grouping the
    # operator terms results in an unpredictable group ordering and, thus, unpredictable circuit to
    # assert against at this point.


def _is_diagonal(term):
    creations = sorted(idx for action, idx in term if action)
    annihilations = sorted(idx for action, idx in term if not action)
    return creations == annihilations


def test_qdrift_optimization_filter_diagonal_terms():
    """Diagonal-term filtering is expected to happen on the Hamiltonian before it is wrapped in an
    Evolution gate (see the QDriftTrotterization class docstring), not on every call to run()."""
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))
    num_modes = 2 * fcidump.norb
    hamil = FermionOperator.from_fcidump(fcidump)
    normal = hamil.normal_ordered().simplify(atol=1e-16)
    group_terms_by_electronic_structure(normal, num_modes, two_body_physicist_order=False)

    # sanity check: the grouped, normal-ordered Hamiltonian still contains diagonal terms (the
    # constant offset and number operators) which the filtering is expected to remove.
    assert any(_is_diagonal(term) for term, _ in normal.iter_terms())

    filter_diagonal_terms(normal)
    assert not any(_is_diagonal(term) for term, _ in normal.iter_terms())

    time = 1.5
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, normal, time=time), circ.modes)

    num_terms = 5
    qdrift = QDriftTrotterization(num_terms, rng=42)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    # none of the sampled sub-operators may contain a diagonal term
    for instruction in qdrift_circ._inner.data:
        for term, _ in instruction.operation.operator.iter_terms():
            assert not _is_diagonal(term), "a diagonal term was sampled despite filtering"


def test_qdrift_optimization_preserves_coefficient_sign():
    """Regression test for a bug where every sampled term's coefficient was replaced with a
    hardcoded +1.0, discarding its sign. This made every rotation point in the same direction
    regardless of whether the original Hamiltonian coefficient was positive or negative, so the
    synthesized circuit did not converge to the target time evolution for Hamiltonians with
    mixed-sign coefficients (the general case).

    Uses a real negative single-body coefficient taken from h2.fcidump (the on-site term for
    mode 0), so this is not merely a synthetic edge case.
    """
    num_modes = 2
    coeff = -1.2563390730032502 + 0j
    hamil = FermionOperator.from_terms([(((True, 0), (False, 0)), coeff)])
    hamil.groups = None

    time = 1.5
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=time), circ.modes)

    num_terms = 3
    qdrift = QDriftTrotterization(num_terms, rng=42)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    # a single-term Hamiltonian means every sampled sub-operator is a copy of that same term;
    # each must retain its negative sign rather than being flattened to +1.0
    for instruction in qdrift_circ._inner.data:
        coeffs = instruction.operation.operator.get_coeffs()
        assert np.all(np.real(coeffs) < 0), f"expected a negative coefficient, got {coeffs}"


def test_qdrift_preserves_non_evolution_gates():
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))
    num_modes = 2 * fcidump.norb
    hamil = FermionOperator.from_fcidump(fcidump)
    hamil.groups = None
    time = 1.5
    circ = FermionicCircuit(num_modes)
    init = InitializeModes([1, 0, 1, 0])
    circ.append(init, circ.modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    num_terms = 4
    qdrift = QDriftTrotterization(num_terms)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"InitializeModes": 1, "Evolution": num_terms}


def test_qdrift_filter_trivial_only_emits_coupling_terms():
    """Every sampled term must have support intersecting both, the (dynamically growing) occupied
    and unoccupied mode sets -- a term entirely confined to one side cannot change the sampled
    bitstring and must never survive the filtering."""
    num_modes = 4
    init = InitializeModes([True, True, False, False])
    hamil = FermionOperator.from_terms(
        [
            (((True, 0), (False, 0)), 1.0),  # n_0: trivial (within occupied)
            (((True, 0), (True, 1), (False, 1), (False, 0)), 1.0),  # n_0 n_1: trivial
            (((True, 2), (False, 0)), 1.0),  # 0 -> 2: couples occupied/unoccupied
            (((True, 3), (False, 1)), 1.0),  # 1 -> 3: couples occupied/unoccupied
            (((True, 3), (False, 2)), 1.0),  # 2 -> 3: trivial (within unoccupied)
        ]
    )
    hamil.groups = None

    circ = FermionicCircuit(num_modes)
    circ.append(init, circ.modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 30
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=42)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"InitializeModes": 1, "Evolution": num_terms}

    occupied = {0, 1}
    unoccupied = {2, 3}
    for instruction in qdrift_circ._inner.data:
        if instruction.operation.name != "Evolution":
            continue
        support = instruction.operation.operator.get_support()
        assert support & occupied and support & unoccupied, (
            f"sampled a trivial term with support {support}"
        )
        occupied |= support
        unoccupied |= support


def test_qdrift_filter_trivial_orbital_rotation_marks_modes_uncertain():
    """An OrbitalRotation between InitializeModes and Evolution mixes creation operators across the
    modes it acts on, so every one of those modes must become "uncertain" (added to both the
    occupied and unoccupied sets) exactly like a mode touched by an accepted qDRIFT term -- a term
    that would otherwise be trivially confined to one side must be accepted once one of its modes
    has been marked uncertain this way."""
    num_modes = 4
    init = InitializeModes([True, True, False, False])  # occupied={0,1}, unoccupied={2,3}
    # mixes mode 0 (occupied) with mode 2 (unoccupied), marking both "uncertain"
    rotation = OrbitalRotation(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))

    # entirely confined to the original occupied set {0, 1} -- trivial unless mode 0 is uncertain
    hamil = FermionOperator.from_terms([(((True, 1), (False, 0)), 1.0)])
    hamil.groups = None

    circ = FermionicCircuit(num_modes)
    circ.append(init, circ.modes)
    circ.append(rotation, [circ.modes[0], circ.modes[2]])
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 5
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=42)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {
        "InitializeModes": 1,
        "OrbitalRotation": 1,
        "Evolution": num_terms,
    }

    for instruction in qdrift_circ._inner.data:
        if instruction.operation.name == "Evolution":
            assert instruction.operation.operator.get_support() == {0, 1}


def test_qdrift_filter_trivial_prepare_slater_determinant_seeds_and_marks_uncertain():
    """PrepareSlaterDeterminant is the composition of an InitializeModes reference occupation and an
    OrbitalRotation (see its class docstring), so filter_trivial must treat it as both applied
    back-to-back: seed the occupied/unoccupied sets from its occupation, then immediately mark every
    mode it acts on as "uncertain" because of the rotation it also carries."""
    num_modes = 4
    # occupation seeds occupied={0,1}, unoccupied={2,3}; the rotation swaps modes 0 and 2 (leaving 1
    # and 3 unchanged), so all four end up "uncertain" right away, just like the OrbitalRotation-only
    # test above (whose rotation is embedded here, extended to the identity on modes 1 and 3).
    rotation_unitary = np.eye(4, dtype=complex)
    rotation_unitary[[0, 2]] = rotation_unitary[[2, 0]]
    prep = PrepareSlaterDeterminant([True, True, False, False], rotation_unitary)

    # entirely confined to the original occupied set {0, 1} -- trivial unless mode 0 is uncertain
    hamil = FermionOperator.from_terms([(((True, 1), (False, 0)), 1.0)])
    hamil.groups = None

    circ = FermionicCircuit(num_modes)
    circ.append(prep, circ.modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 5
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=42)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {
        "PrepareSlaterDeterminant": 1,
        "Evolution": num_terms,
    }

    for instruction in qdrift_circ._inner.data:
        if instruction.operation.name == "Evolution":
            assert instruction.operation.operator.get_support() == {0, 1}


def test_qdrift_filter_trivial_rejects_purely_diagonal_hamiltonian():
    """When every term is diagonal (i.e. never couples the occupied/unoccupied sets), filtering can
    never find a non-trivial term, so sampling must exhaust its retry budget and raise."""
    num_modes = 2
    init = InitializeModes([True, False])
    hamil = FermionOperator.from_terms([(((True, 0), (False, 0)), 1.0)])
    hamil.groups = None

    circ = FermionicCircuit(num_modes)
    circ.append(init, circ.modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    qdrift = QDriftTrotterization(3, filter_trivial=True, rng=1)
    qdrift.MAX_SAMPLE_RETRIES = 100
    pm = FermionicPassManager(qdrift)

    with pytest.raises(RuntimeError, match="non-trivial term"):
        pm.run(circ)


def test_qdrift_filter_trivial_warns_without_initialize_modes():
    """filter_trivial requires occupation information from a preceding InitializeModes gate; without
    one, filtering cannot be applied and a UserWarning must be emitted instead of silently ignoring
    the flag or raising."""
    num_modes = 2
    hamil = FermionOperator.from_terms([(((True, 0), (False, 1)), 1.0)])
    hamil.groups = None

    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 3
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=1)
    pm = FermionicPassManager(qdrift)

    with pytest.warns(UserWarning, match="not preceded by an InitializeModes gate"):
        qdrift_circ = pm.run(circ)

    assert qdrift_circ.count_ops() == {"Evolution": num_terms}


def test_qdrift_filter_trivial_warns_when_all_modes_occupied():
    """When the preceding InitializeModes gate(s) mark every mode as occupied, there is no
    unoccupied mode left to couple against, so filtering cannot be applied and a UserWarning must
    be emitted instead of silently ignoring the flag or raising."""
    num_modes = 2
    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes([True, True]), circ.modes)

    hamil = FermionOperator.from_terms([(((True, 0), (False, 1)), 1.0)])
    hamil.groups = None
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 3
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=1)
    pm = FermionicPassManager(qdrift)

    with pytest.warns(UserWarning, match="marked every mode as occupied"):
        qdrift_circ = pm.run(circ)

    assert qdrift_circ.count_ops() == {"InitializeModes": 1, "Evolution": num_terms}


def test_qdrift_filter_trivial_warns_when_all_modes_unoccupied():
    """When the preceding InitializeModes gate(s) mark every mode as unoccupied, there is no
    occupied mode left to couple against, so filtering cannot be applied and a UserWarning must be
    emitted instead of silently ignoring the flag or raising."""
    num_modes = 2
    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes([False, False]), circ.modes)

    hamil = FermionOperator.from_terms([(((True, 0), (False, 1)), 1.0)])
    hamil.groups = None
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 3
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=1)
    pm = FermionicPassManager(qdrift)

    with pytest.warns(UserWarning, match="marked every mode as unoccupied"):
        qdrift_circ = pm.run(circ)

    assert qdrift_circ.count_ops() == {"InitializeModes": 1, "Evolution": num_terms}


def test_qdrift_filter_trivial_accumulates_parallel_initialize_modes():
    """Several InitializeModes gates placed in parallel (e.g. one per spin sector) must have their
    occupation information accumulated together, correctly mapped onto *global* mode indices rather
    than assumed to start at index 0."""
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    # alpha sector (modes 0, 1): mode 0 occupied, mode 1 unoccupied
    circ.append(InitializeModes([True, False]), [circ.modes[0], circ.modes[1]])
    # beta sector (modes 2, 3): mode 2 unoccupied, mode 3 occupied
    circ.append(InitializeModes([False, True]), [circ.modes[2], circ.modes[3]])

    hamil = FermionOperator.from_terms(
        [
            (((True, 1), (False, 0)), 1.0),  # 0 -> 1: occupied -> unoccupied (non-trivial)
            (((True, 2), (False, 3)), 1.0),  # 3 -> 2: occupied -> unoccupied (non-trivial)
        ]
    )
    hamil.groups = None
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 10
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=7)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"InitializeModes": 2, "Evolution": num_terms}

    # both terms already couple occupied with unoccupied modes from the start, so every sample must
    # be accepted immediately -- exercising this confirms the occupation was read from the correct
    # (global) mode indices rather than e.g. both gates being wrongly assumed to start at 0.
    for instruction in qdrift_circ._inner.data:
        if instruction.operation.name == "Evolution":
            assert instruction.operation.operator.get_support() in ({0, 1}, {2, 3})


def _evolutions(circuit: FermionicCircuit) -> list[Evolution]:
    return [
        instruction.operation
        for instruction in circuit._inner.data
        if isinstance(instruction.operation, Evolution)
    ]


def _coupling_hamiltonian() -> FermionOperator:
    """A 4-mode Hamiltonian whose terms all couple modes 0/1 with modes 2/3.

    Every term bridges the occupied and unoccupied sets seeded by ``InitializeModes([1, 1, 0, 0])``,
    so it is usable by the ``filter_trivial=True`` path without any draw being rejected.
    """
    hamil = FermionOperator.from_terms(
        [
            (((True, 2), (False, 0)), 1.0),  # 0 -> 2
            (((True, 3), (False, 1)), 1.0),  # 1 -> 3
            (((True, 0), (False, 2)), 1.0),  # 2 -> 0
            (((True, 1), (False, 3)), 1.0),  # 3 -> 1
        ]
    )
    hamil.groups = None
    return hamil


def _filter_trivial_circuit(hamil: FermionOperator, **kwargs) -> FermionicCircuit:
    """Builds a circuit whose Evolution is preceded by the InitializeModes ``filter_trivial`` needs."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([True, True, False, False]), circ.modes)
    circ.append(Evolution(4, hamil, time=1.0, **kwargs), circ.modes)
    return circ


def test_qdrift_forwards_the_synthesis_method(subtests):
    """The synthesis method of the input gate must survive the pass.

    Dropping it silently replaced a caller's choice with the default ``FermionicLieTrotter``. Both
    emission paths construct their gate independently, so both are covered here.
    """
    synthesis = FermionicSuzukiTrotter(order=2, reps=3)
    num_terms = 4

    with subtests.test("batched sampling"):
        circ = _filter_trivial_circuit(_coupling_hamiltonian(), synthesis=synthesis)

        qdrift_circ = FermionicPassManager(QDriftTrotterization(num_terms, rng=42)).run(circ)

        evolutions = _evolutions(qdrift_circ)
        assert len(evolutions) == num_terms
        for evolution in evolutions:
            assert evolution.synthesis is synthesis

    with subtests.test("rejection sampling"):
        circ = _filter_trivial_circuit(_coupling_hamiltonian(), synthesis=synthesis)

        qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=42)
        qdrift_circ = FermionicPassManager(qdrift).run(circ)

        evolutions = _evolutions(qdrift_circ)
        assert len(evolutions) == num_terms
        for evolution in evolutions:
            assert evolution.synthesis is synthesis


def test_qdrift_gates_are_atomic(subtests):
    """The sampled gates are terminal factors: the random draw *is* the Trotterization.

    Decomposing them again would discard the sampling this pass performed.
    """
    num_terms = 4

    with subtests.test("batched sampling"):
        circ = _filter_trivial_circuit(_coupling_hamiltonian())

        qdrift_circ = FermionicPassManager(QDriftTrotterization(num_terms, rng=42)).run(circ)

        evolutions = _evolutions(qdrift_circ)
        assert len(evolutions) == num_terms
        assert all(evolution.atomic for evolution in evolutions)

    with subtests.test("rejection sampling"):
        circ = _filter_trivial_circuit(_coupling_hamiltonian())

        qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=42)
        qdrift_circ = FermionicPassManager(qdrift).run(circ)

        evolutions = _evolutions(qdrift_circ)
        assert len(evolutions) == num_terms
        assert all(evolution.atomic for evolution in evolutions)


def test_qdrift_does_not_mutate_the_input_gate():
    """The pass must build new gates rather than mutate the node's operation.

    A gate instance can be shared between circuits (Qiskit copies gates with a shallow ``__dict__``
    copy), so mutating one would retroactively change every other circuit holding the same object.
    """
    gate = Evolution(4, _coupling_hamiltonian(), time=1.0)
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([True, True, False, False]), circ.modes)
    circ.append(gate, circ.modes)

    FermionicPassManager(QDriftTrotterization(4, rng=42)).run(circ)

    assert not gate.atomic


def test_qdrift_output_is_a_decomposition_fixed_point():
    """Decomposing the sampled gates must not split them into non-unitary factors.

    A sampled *group* holds several terms, and ``split_out_groups`` drops the grouping, so before the
    emitted gates were marked atomic a further decomposition re-split each group term by term. An
    individual term is generally not Hermitian even when its group is (the conjugate pairs of a UCC
    cluster generator being the motivating example), so the exponential of such a factor is not
    unitary, and the fermion-to-qubit stage rejected it with a ``ValueError`` about complex
    coefficients.
    """
    num_modes = 4
    hamil = FermionOperator.from_terms(
        [
            (((True, 0), (False, 1)), 1.0j),  # group 0: Hermitian as a pair,
            (((True, 1), (False, 0)), -1.0j),  #          but neither term is on its own
            (((True, 2), (False, 3)), 1.0j),  # group 1: likewise
            (((True, 3), (False, 2)), -1.0j),
        ]
    )
    hamil.groups = [0, 0, 1, 1]

    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 2
    qdrift_circ = FermionicPassManager(QDriftTrotterization(num_terms, rng=5)).run(circ)

    assert qdrift_circ.count_ops() == {"Evolution": num_terms}
    # repeated decomposition makes no further progress ...
    assert qdrift_circ.decompose(reps=5).count_ops() == {"Evolution": num_terms}
    # ... so every factor keeps the Hermiticity of the group it was sampled from
    for evolution in _evolutions(qdrift_circ.decompose(reps=5)):
        operator = evolution.operator
        assert operator.adjoint().equiv(operator), f"non-Hermitian factor {operator}"


def _ungrouped_hamiltonian() -> FermionOperator:
    """A 4-mode Hamiltonian with mixed-sign, non-uniform coefficients and no groups."""
    hamil = FermionOperator.from_terms(
        [
            (((True, 0), (False, 1)), 0.7),
            (((True, 1), (False, 0)), 0.7),
            (((True, 1), (False, 2)), -0.4),
            (((True, 2), (False, 1)), -0.4),
        ]
    )
    hamil.groups = None
    return hamil


def _weights_circuit(hamil: FermionOperator, time: float = 1.5) -> FermionicCircuit:
    circ = FermionicCircuit(4)
    circ.append(Evolution(4, hamil, time=time), circ.modes)
    return circ


def test_qdrift_explicit_default_weights_are_a_no_op():
    """Passing the weights the pass would have computed itself must change nothing.

    This pins the equivalence the ``weights`` option relies on: ``|c_j|`` reproduces the textbook
    distribution *and* the textbook ``delta``, so a caller hoisting the weight computation out of an
    ensemble loop gets bit-identical circuits rather than merely statistically similar ones.
    """
    hamil = _ungrouped_hamiltonian()
    weights = np.abs(np.array(hamil.get_coeffs()))

    implicit = FermionicPassManager(QDriftTrotterization(4, rng=42)).run(_weights_circuit(hamil))
    explicit = FermionicPassManager(QDriftTrotterization(4, rng=42, weights=weights)).run(
        _weights_circuit(hamil)
    )

    for lhs, rhs in zip(_evolutions(implicit), _evolutions(explicit), strict=True):
        assert lhs.operator.equiv(rhs.operator)
        assert lhs.params[0] == rhs.params[0]


def test_qdrift_weights_scale_sets_the_evolution_time():
    """The weights are absolute magnitudes, not relative preferences.

    ``delta * p_j == |w_j| * t / num_terms`` depends on ``w_j`` itself and not merely on its share of
    the 1-norm, so scaling every weight by ``gamma`` samples identically but evolves for ``gamma * t``.
    Asserted exactly, since it follows from the arithmetic rather than from the draws.
    """
    hamil = _ungrouped_hamiltonian()
    weights = np.abs(np.array(hamil.get_coeffs()))
    num_terms = 4

    baseline = FermionicPassManager(QDriftTrotterization(num_terms, rng=42, weights=weights)).run(
        _weights_circuit(hamil)
    )
    scaled = FermionicPassManager(
        QDriftTrotterization(num_terms, rng=42, weights=2.0 * weights)
    ).run(_weights_circuit(hamil))

    for lhs, rhs in zip(_evolutions(baseline), _evolutions(scaled), strict=True):
        # the same draws (the distribution is unchanged) for exactly twice as long
        assert lhs.operator.equiv(rhs.operator)
        assert rhs.params[0] == pytest.approx(2.0 * lhs.params[0])

    # and the absolute value follows the documented formula
    expected = (np.abs(weights).sum() * 1.5) / num_terms
    assert all(
        evolution.params[0] == pytest.approx(expected) for evolution in _evolutions(baseline)
    )


def test_qdrift_weights_change_the_sampled_distribution():
    """A reshaped distribution must actually redirect the draws.

    ``_coupling_hamiltonian`` has uniform coefficients, so the default distribution is uniform too;
    concentrating all the weight on a single term must make every draw that term.
    """
    hamil = _coupling_hamiltonian()
    weights = np.array([1.0, 0.0, 0.0, 0.0])

    qdrift_circ = FermionicPassManager(QDriftTrotterization(5, rng=42, weights=weights)).run(
        _weights_circuit(hamil)
    )

    evolutions = _evolutions(qdrift_circ)
    assert len(evolutions) == 5
    only = FermionOperator.from_terms([(((True, 2), (False, 0)), 1.0)])
    for evolution in evolutions:
        assert evolution.operator.equiv(only)


def test_qdrift_weights_reject_a_path_shaped_array():
    """An array of length ``num_terms ** M`` must be rejected rather than sampled incorrectly.

    A signed quasi-probability distribution over length-``M`` *paths* has one entry per path, that is
    ``num_terms ** M`` of them, and is the planned generalization of this argument. Until it is
    supported, such an array has to fail the length check instead of being consumed as if it held one
    weight per term.
    """
    hamil = _ungrouped_hamiltonian()
    num_terms = len(hamil)

    for slices in (2, 3):
        paths = np.ones(num_terms**slices)
        with pytest.raises(ValueError, match="exactly one entry per sampled piece"):
            FermionicPassManager(QDriftTrotterization(4, weights=paths)).run(
                _weights_circuit(hamil)
            )


def test_qdrift_weights_length_must_match_the_operator(subtests):
    """One entry per sampled piece: per group when the operator is grouped, per term otherwise."""
    with subtests.test("ungrouped, too few"):
        qdrift = QDriftTrotterization(4, weights=np.ones(3))
        with pytest.raises(ValueError, match="one entry per sampled piece"):
            FermionicPassManager(qdrift).run(_weights_circuit(_ungrouped_hamiltonian()))

    with subtests.test("ungrouped, too many"):
        qdrift = QDriftTrotterization(4, weights=np.ones(5))
        with pytest.raises(ValueError, match="one entry per sampled piece"):
            FermionicPassManager(qdrift).run(_weights_circuit(_ungrouped_hamiltonian()))

    with subtests.test("grouped counts groups, not terms"):
        hamil = _ungrouped_hamiltonian()
        hamil.groups = [0, 0, 1, 1]
        # four terms but only two groups, so a per-term array is rejected ...
        with pytest.raises(ValueError, match="2 groups"):
            FermionicPassManager(QDriftTrotterization(4, weights=np.ones(4))).run(
                _weights_circuit(hamil)
            )
        # ... while a per-group one is accepted
        qdrift_circ = FermionicPassManager(QDriftTrotterization(4, rng=42, weights=np.ones(2))).run(
            _weights_circuit(hamil)
        )
        assert qdrift_circ.count_ops() == {"Evolution": 4}


def test_qdrift_weights_reject_several_evolution_gates():
    """A weights array describes one specific operator, so a circuit holding several Evolution gates
    is ambiguous and must be rejected rather than sampled against the wrong operator. Without custom
    weights the same circuit stays supported, since each gate derives its own."""
    hamil = _ungrouped_hamiltonian()
    circ = FermionicCircuit(4)
    circ.append(Evolution(4, hamil, time=1.5), circ.modes)
    circ.append(Evolution(4, hamil, time=1.5), circ.modes)

    weights = np.abs(np.array(hamil.get_coeffs()))
    with pytest.raises(ValueError, match="more than one Evolution gate"):
        FermionicPassManager(QDriftTrotterization(3, rng=42, weights=weights)).run(circ)

    qdrift_circ = FermionicPassManager(QDriftTrotterization(3, rng=42)).run(circ)
    assert qdrift_circ.count_ops() == {"Evolution": 6}


def test_qdrift_rejects_invalid_weights(subtests):
    """Malformed weights are rejected at construction, before any circuit is seen."""
    with subtests.test("empty"), pytest.raises(ValueError, match="must not be empty"):
        QDriftTrotterization(4, weights=[])

    with subtests.test("multi-dimensional"), pytest.raises(ValueError, match="one-dimensional"):
        QDriftTrotterization(4, weights=np.ones((2, 2)))

    with subtests.test("non-finite"):
        with pytest.raises(ValueError, match="finite"):
            QDriftTrotterization(4, weights=[1.0, np.nan, 1.0])
        with pytest.raises(ValueError, match="finite"):
            QDriftTrotterization(4, weights=[1.0, np.inf, 1.0])

    with subtests.test("negative entries"):
        # A weight is the magnitude of the qDRIFT decomposition, whose sign belongs to the sampled
        # operator instead; a signed (quasi-probability) distribution needs post-processing support
        # this pass does not provide, so it must be refused rather than silently used.
        with pytest.raises(ValueError, match="Negative sampling weights"):
            QDriftTrotterization(4, weights=[1.0, -1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="Negative sampling weights"):
            QDriftTrotterization(4, weights=-np.ones(4))

    with subtests.test("vanishing sum"), pytest.raises(ValueError, match="must not sum to zero"):
        QDriftTrotterization(4, weights=np.zeros(4))


def test_qdrift_weights_accepts_a_plain_sequence():
    """A list is as good as an array; it is converted once at construction."""
    qdrift = QDriftTrotterization(4, rng=42, weights=[0.7, 0.7, 0.4, 0.4])
    assert isinstance(qdrift.weights, np.ndarray)

    qdrift_circ = FermionicPassManager(qdrift).run(_weights_circuit(_ungrouped_hamiltonian()))
    assert qdrift_circ.count_ops() == {"Evolution": 4}


def _dense_matrix(operator: FermionOperator, num_modes: int) -> np.ndarray:
    """Maps ``operator`` to qubits and materializes it densely.

    Deliberately routed through :func:`.jordan_wigner` rather than through a simulation backend, so
    that this needs no optional dependency.
    """
    return SparsePauliOp.from_sparse_observable(jordan_wigner(operator, num_modes)).to_matrix()


def _ensemble_mean(circuit_factory, num_samples: int, num_modes: int) -> np.ndarray:
    """Averages the unitary implemented by ``num_samples`` randomizations.

    The per-factor propagators are memoized on the sampled operator, since a randomization draws
    repeatedly from the same handful of terms and each ``expm`` is far costlier than the lookup.
    """
    dim = 2**num_modes
    accumulated = np.zeros((dim, dim), dtype=complex)
    propagators: dict[tuple, np.ndarray] = {}

    for _ in range(num_samples):
        product = np.eye(dim, dtype=complex)
        for evolution in _evolutions(circuit_factory()):
            key = (
                tuple(
                    (tuple(actions), coeff) for actions, coeff in evolution.operator.iter_terms()
                ),
                evolution.params[0],
            )
            propagator = propagators.get(key)
            if propagator is None:
                factor = _dense_matrix(evolution.operator, num_modes)
                propagator = expm(-1j * evolution.params[0] * factor)
                propagators[key] = propagator
            product = propagator @ product
        accumulated += product

    return accumulated / num_samples


def test_qdrift_ensemble_mean_approximates_the_target_evolution(subtests):
    """The *ensemble* of randomizations is what approximates ``exp(-i t H)``, and a rescaled weights
    array retargets it onto ``exp(-i (gamma t) H)``.

    A single randomization is a crude approximation, so this averages over many and checks that the
    error *shrinks* as ``num_terms`` grows. The tolerances are deliberately loose: the residual mixes
    the Trotter error (falling as ``1 / num_terms``) with Monte-Carlo noise (falling only as
    ``1 / sqrt(num_samples)``), so this is a guard against a gross sign or normalization inversion
    rather than a measurement of accuracy. The exact arithmetic is pinned by
    ``test_qdrift_weights_scale_sets_the_evolution_time`` instead.
    """
    num_modes = 4
    hamil = FermionOperator.from_terms(
        [
            (((True, 0), (False, 1)), 0.7),
            (((True, 1), (False, 0)), 0.7),
            (((True, 1), (False, 2)), -0.4),
            (((True, 2), (False, 1)), -0.4),
        ]
    )
    hamil.groups = None
    time = 0.25
    weights = np.abs(np.array(hamil.get_coeffs()))
    matrix = _dense_matrix(hamil, num_modes)
    num_samples = 400

    def sampler(num_terms, sampling_weights):
        qdrift = QDriftTrotterization(num_terms, rng=42, weights=sampling_weights)
        pass_manager = FermionicPassManager(qdrift)
        return lambda: pass_manager.run(_weights_circuit(hamil, time=time))

    with subtests.test("converges towards the exact evolution"):
        target = expm(-1j * time * matrix)
        coarse_error = np.linalg.norm(
            _ensemble_mean(sampler(5, weights), num_samples, num_modes) - target, 2
        )
        fine_error = np.linalg.norm(
            _ensemble_mean(sampler(40, weights), num_samples, num_modes) - target, 2
        )
        assert fine_error < coarse_error, f"error grew: {coarse_error} -> {fine_error}"
        assert fine_error < 5e-2

    with subtests.test("rescaled weights retarget the evolution time"):
        # 2x the weights evolves for 2t, so the SAME ensemble is close to exp(-i (2t) H) and far
        # from exp(-i t H) -- which is what makes the scale a physical parameter, not a gauge.
        mean = _ensemble_mean(sampler(40, 2.0 * weights), num_samples, num_modes)
        towards_scaled = np.linalg.norm(mean - expm(-1j * (2.0 * time) * matrix), 2)
        towards_unscaled = np.linalg.norm(mean - expm(-1j * time * matrix), 2)
        assert towards_scaled < 5e-2
        assert towards_scaled < towards_unscaled


def test_qdrift_filter_trivial_records_discarded_metadata():
    """Filtering must report how many draws it rejected, since the acceptance rate is what
    quantifies the bias the rejection introduces. The counts land in the output DAG's metadata, one
    entry per filtered Evolution gate.

    The Hamiltonian below is weighted so that rejection is unavoidable: the two trivial terms carry
    almost all of the coefficient magnitude, so the first draw is overwhelmingly likely to be one of
    them, and on the first draw the tracked sets are still tight enough to reject it. Note that a
    Hamiltonian with balanced weights can legitimately discard nothing, because an early accepted
    term marks the modes it touches "uncertain" and thereby makes the remaining terms acceptable.
    """
    num_modes = 4
    hamil = FermionOperator.from_terms(
        [
            (((True, 0), (False, 0)), 1.0e4),  # n_0: trivial (within occupied)
            (((True, 3), (False, 2)), 1.0e4),  # 2 -> 3: trivial (within unoccupied)
            (((True, 2), (False, 0)), 1.0),  # 0 -> 2: couples occupied/unoccupied
        ]
    )
    hamil.groups = None

    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes([True, True, False, False]), circ.modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 5
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=42)

    qdrift_circ = FermionicPassManager(qdrift).run(circ)

    assert qdrift_circ.metadata["filter_trivial.emitted"] == [num_terms]
    # The exact count depends on the rng stream and must not be pinned; that it is positive is what
    # proves the rejections are being counted.
    assert qdrift_circ.metadata["filter_trivial.discarded"][0] > 0


def test_qdrift_filter_trivial_metadata_absent_when_not_filtering():
    """The metadata is only written when filtering actually ran, so callers must read it
    defensively. Neither a disabled flag nor a skipped (warned) gate may leave the fields
    behind."""
    hamil = _coupling_hamiltonian()

    with_flag_off = FermionicPassManager(QDriftTrotterization(3, rng=1)).run(
        _filter_trivial_circuit(hamil)
    )
    assert "filter_trivial.discarded" not in with_flag_off.metadata
    assert "filter_trivial.emitted" not in with_flag_off.metadata

    # A gate whose filtering is skipped (no InitializeModes to seed against) must not record either.
    circ = FermionicCircuit(4)
    circ.append(Evolution(4, hamil, time=1.0), circ.modes)
    qdrift = QDriftTrotterization(3, filter_trivial=True, rng=1)
    with pytest.warns(UserWarning, match="not preceded by an InitializeModes gate"):
        skipped = FermionicPassManager(qdrift).run(circ)
    assert "filter_trivial.discarded" not in skipped.metadata
    assert "filter_trivial.emitted" not in skipped.metadata


def test_qdrift_filter_trivial_metadata_records_zero_discards():
    """A discarded count of zero must be distinguishable from an absent field: it says the filtering
    ran and accepted every draw, leaving the sampling distribution untouched, which is exactly the
    case a user reading the diagnostic wants to see."""
    hamil = _coupling_hamiltonian()  # every term couples, so nothing gets rejected
    qdrift = QDriftTrotterization(4, filter_trivial=True, rng=1)

    qdrift_circ = FermionicPassManager(qdrift).run(_filter_trivial_circuit(hamil))

    assert qdrift_circ.metadata["filter_trivial.discarded"] == [0]
    assert qdrift_circ.metadata["filter_trivial.emitted"] == [4]


def test_qdrift_filter_trivial_metadata_is_per_gate():
    """The counts are recorded per Evolution gate rather than summed, because the tracked mode sets
    carry across gates and the rejection bites hardest on the earliest draws."""
    num_modes = 4
    hamil = _coupling_hamiltonian()

    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes([True, True, False, False]), circ.modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)
    circ.append(Evolution(num_modes, hamil, time=1.0), circ.modes)

    num_terms = 3
    qdrift = QDriftTrotterization(num_terms, filter_trivial=True, rng=42)

    qdrift_circ = FermionicPassManager(qdrift).run(circ)

    assert qdrift_circ.metadata["filter_trivial.emitted"] == [num_terms, num_terms]
    assert len(qdrift_circ.metadata["filter_trivial.discarded"]) == 2


def test_qdrift_filter_trivial_metadata_pickle():
    """Circuits carrying the filtering diagnostics must remain serializable, mirroring the guard put
    in place for the RelabelModes metadata (see qiskit-fermions issue #225)."""
    qdrift = QDriftTrotterization(3, filter_trivial=True, rng=42)
    circ = _filter_trivial_circuit(_coupling_hamiltonian())

    qdrift_circ = FermionicPassManager(qdrift).run(circ)
    assert "filter_trivial.emitted" in qdrift_circ.metadata

    reconstructed = pickle.loads(pickle.dumps(qdrift_circ._inner))
    for key in ("filter_trivial.discarded", "filter_trivial.emitted"):
        assert reconstructed.metadata[key] == qdrift_circ.metadata[key]
