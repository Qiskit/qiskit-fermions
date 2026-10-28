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

"""Tests for the deprecated signature of ``QDriftTrotterization``."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes
from qiskit_fermions.circuit.library.synthesis import FermionicQDrift
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicPassManager
from qiskit_fermions.transpiler.passes import F2FSynthesis, QDriftTrotterization


def _operator() -> FermionOperator:
    return FermionOperator.from_terms_with_groups(
        [
            (((True, 0), (False, 1)), -1.0, 0),
            (((True, 1), (False, 0)), -1.0, 0),
            (((True, 1), (False, 2)), -0.8, 1),
            (((True, 2), (False, 1)), -0.8, 1),
            (((True, 2), (False, 3)), -0.5, 2),
            (((True, 3), (False, 2)), -0.5, 2),
        ]
    )


def _circuit() -> FermionicCircuit:
    circuit = FermionicCircuit(4)
    circuit.append(InitializeModes([1, 1, 0, 0]), circuit.modes)
    circuit.append(Evolution(4, _operator(), time=0.5), circuit.modes)
    return circuit


def _times_and_supports(circuit):
    return [
        (float(inst.operation.params[0]), sorted(inst.operation.operator.get_support()))
        for inst in circuit._inner.data
        if inst.operation.name == "Evolution"
    ]


def test_the_deprecation_is_announced():
    with pytest.warns(DeprecationWarning, match="FermionicQDrift"):
        QDriftTrotterization(3, rng=1)


def test_the_new_signature_does_not_warn(recwarn):
    QDriftTrotterization(FermionicQDrift(3, rng=1))

    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_the_old_attributes_are_kept():
    weights = np.array([1.0, 1.0, 1.0])
    with pytest.warns(DeprecationWarning):
        old = QDriftTrotterization(3, filter_trivial=True, weights=weights, rng=1)

    assert isinstance(old, QDriftTrotterization)
    assert (old.num_terms, old.filter_trivial) == (3, True)
    assert old.weights is not None
    assert old.weights.tolist() == weights.tolist()


def test_without_filtering_it_samples_as_the_method_does():
    with pytest.warns(DeprecationWarning):
        old = QDriftTrotterization(5, rng=7)
    new = F2FSynthesis({"Evolution": FermionicQDrift(5, rng=7)})

    a = FermionicPassManager(old).run(_circuit())
    b = FermionicPassManager(new).run(_circuit())

    assert _times_and_supports(a) == _times_and_supports(b)


def test_with_filtering_it_matches_the_pass():
    with pytest.warns(DeprecationWarning):
        old = QDriftTrotterization(5, filter_trivial=True, rng=7)
    new = QDriftTrotterization(FermionicQDrift(5, rng=7))

    a = FermionicPassManager(old).run(_circuit())
    b = FermionicPassManager(new).run(_circuit())

    assert _times_and_supports(a) == _times_and_supports(b)
    assert a.metadata == b.metadata
