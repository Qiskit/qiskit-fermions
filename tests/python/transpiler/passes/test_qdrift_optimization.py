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
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.grouping import group_terms_by_electronic_structure
from qiskit_fermions.operators.library import FCIDump
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
