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

from pathlib import Path

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.circuit.library.synthesis import FermionicSuzukiTrotter
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.library import FCIDump
from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms
from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
from qiskit_fermions.transpiler.passes import QDriftTrotterization
from qiskit_fermions.transpiler.passmanager import FermionicPassManager


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
