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
from qiskit_fermions.circuit.library import Evolution, InitializeModes
from qiskit_fermions.operators import FermionOperator, MajoranaOperator, gamma
from qiskit_fermions.operators.library import FCIDump
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
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))
    num_modes = 2 * fcidump.norb
    hamil = FermionOperator.from_fcidump(fcidump)
    normal = hamil.normal_ordered().simplify(atol=1e-16)
    group_terms_by_electronic_structure(normal, num_modes, two_body_physicist_order=False)

    # sanity check: the grouped, normal-ordered Hamiltonian still contains diagonal terms (the
    # constant offset and number operators) which the filtering is expected to remove.
    assert any(_is_diagonal(term) for term, _ in normal.iter_terms())
    num_groups_before = normal.num_groups()

    time = 1.5
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, normal, time=time), circ.modes)

    num_terms = 5
    qdrift = QDriftTrotterization(num_terms, filter_diagonal_terms=True, rng=42)
    pm = FermionicPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    # the user's original operator must not have been mutated by the pass
    assert normal.num_groups() == num_groups_before
    assert any(_is_diagonal(term) for term, _ in normal.iter_terms())

    # none of the sampled sub-operators may contain a diagonal term
    for instruction in qdrift_circ._inner.data:
        for term, _ in instruction.operation.operator.iter_terms():
            assert not _is_diagonal(term), "a diagonal term was sampled despite filtering"


def test_qdrift_optimization_filter_diagonal_terms_rejects_non_fermion_operator():
    """Diagonal-term filtering is only defined for fermionic number operators, so requesting it for
    an Evolution gate carrying a non-FermionOperator must raise TypeError rather than passing the
    wrong operator type into the fermion-specific filter."""
    num_modes = 2
    hamil = MajoranaOperator.from_dict({(gamma(0, False), gamma(1, False)): 1.0})

    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=1.5), circ.modes)

    qdrift = QDriftTrotterization(5, filter_diagonal_terms=True, rng=42)
    pm = FermionicPassManager(qdrift)

    with pytest.raises(TypeError, match="only supported for Evolution gates"):
        pm.run(circ)


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
