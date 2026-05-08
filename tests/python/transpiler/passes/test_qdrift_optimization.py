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
from qiskit_fermions.circuit import FermionCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.grouping import group_terms_by_electronic_structure
from qiskit_fermions.operators.library import FCIDump
from qiskit_fermions.transpiler.passes import QDriftTrotterization
from qiskit_fermions.transpiler.passmanager import FermionPassManager


def test_qdrift_optimization_no_groups(subtests):
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))
    num_fermions = 2 * fcidump.norb
    hamil = FermionOperator.from_fcidump(fcidump)
    time = 1.5
    circ = FermionCircuit(num_fermions)
    evo = Evolution(num_fermions, hamil, time=time)
    circ.append(evo, circ.fermions)

    with subtests.test("num_terms=4"):
        num_terms = 4
        qdrift = QDriftTrotterization(num_terms)
        pm = FermionPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    with subtests.test("num_terms=6"):
        num_terms = 6
        qdrift = QDriftTrotterization(num_terms)
        pm = FermionPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    with subtests.test("rng seed"):
        num_terms = 2
        qdrift = QDriftTrotterization(num_terms, rng=42)
        pm = FermionPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

        expected_gates = [
            Evolution(
                num_fermions,
                FermionOperator.from_terms(
                    [(((True, 0), (True, 1), (False, 1), (False, 0)), 0.3322908651276483)]
                ),
                time=8.273087572037902,
            ),
            Evolution(
                num_fermions,
                FermionOperator.from_terms(
                    [(((True, 2), (True, 0), (False, 0), (False, 2)), 0.33785507740175824)]
                ),
                time=8.273087572037902,
            ),
        ]

        for actual, expected in zip(qdrift_circ._inner.data, expected_gates, strict=True):
            assert actual.operation.operator.equiv(expected.operator)
            assert np.isclose(actual.params[0], expected.params[0])

    with subtests.test("rng seed"):
        num_terms = 2
        qdrift = QDriftTrotterization(num_terms, rng=np.random.default_rng(43))
        pm = FermionPassManager(qdrift)

        qdrift_circ = pm.run(circ)
        assert qdrift_circ.count_ops() == {"Evolution": num_terms}

        expected_gates = [
            Evolution(
                num_fermions,
                FermionOperator.from_terms(
                    [(((True, 1), (True, 0), (False, 0), (False, 1)), 0.3322908651276483)]
                ),
                time=8.273087572037902,
            ),
            Evolution(
                num_fermions,
                FermionOperator.from_terms([((), 0.7199689944489797)]),
                time=8.273087572037902,
            ),
        ]

        for actual, expected in zip(qdrift_circ._inner.data, expected_gates, strict=True):
            assert actual.operation.operator.equiv(expected.operator)
            assert np.isclose(actual.params[0], expected.params[0])


def test_qdrift_optimization_with_groups():
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))
    num_fermions = 2 * fcidump.norb
    hamil = FermionOperator.from_fcidump(fcidump)
    group_terms_by_electronic_structure(hamil, num_fermions)
    time = 1.5
    circ = FermionCircuit(num_fermions)
    evo = Evolution(num_fermions, hamil, time=time)
    circ.append(evo, circ.fermions)

    num_terms = 2
    qdrift = QDriftTrotterization(num_terms, rng=42)
    pm = FermionPassManager(qdrift)

    qdrift_circ = pm.run(circ)
    assert qdrift_circ.count_ops() == {"Evolution": num_terms}

    expected_gates = [
        Evolution(
            num_fermions,
            FermionOperator.from_terms(
                [
                    (((True, 1), (True, 2), (False, 3), (False, 0)), 0.09046559989211567),
                    (((True, 3), (True, 0), (False, 1), (False, 2)), 0.09046559989211567),
                ]
            ),
            time=6.036695974299787,
        ),
        Evolution(
            num_fermions,
            FermionOperator.from_terms([(((True, 1), (False, 1)), -0.4718960072811406)]),
            time=6.036695974299787,
        ),
    ]

    for actual, expected in zip(qdrift_circ._inner.data, expected_gates, strict=True):
        assert actual.operation.operator.equiv(expected.operator)
        assert np.isclose(actual.params[0], expected.params[0])
