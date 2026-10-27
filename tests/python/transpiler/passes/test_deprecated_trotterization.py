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

"""Tests for the deprecated ``FermionicTrotterization`` shim."""

from __future__ import annotations

import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.circuit.library.synthesis import FermionicSuzukiTrotter
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import FermionicPassManager
from qiskit_fermions.transpiler.passes import F2FSynthesis, FermionicTrotterization


def _circuit() -> FermionicCircuit:
    operator = FermionOperator.from_terms_with_groups(
        [
            (((True, 0), (False, 1)), -1.0, 0),
            (((True, 1), (False, 0)), -1.0, 0),
            (((True, 1), (False, 2)), -0.8, 1),
            (((True, 2), (False, 1)), -0.8, 1),
        ]
    )
    circuit = FermionicCircuit(3)
    circuit.append(Evolution(3, operator, time=0.5), circuit.modes)
    return circuit


def test_the_deprecation_is_announced():
    with pytest.warns(DeprecationWarning, match="F2FSynthesis"):
        FermionicTrotterization(FermionicSuzukiTrotter(order=2))


def test_it_is_an_f2f_synthesis_of_the_evolution():
    method = FermionicSuzukiTrotter(order=2)

    with pytest.warns(DeprecationWarning):
        shim = FermionicTrotterization(method)

    assert isinstance(shim, F2FSynthesis)
    assert shim.methods == {"Evolution": method}


@pytest.mark.parametrize("apply", [True, False])
def test_it_matches_the_pass_it_forwards_to(apply):
    method = FermionicSuzukiTrotter(order=2, reps=2)
    with pytest.warns(DeprecationWarning):
        shim = FermionicTrotterization(method, apply=apply)

    old = FermionicPassManager(shim).run(_circuit())
    new = FermionicPassManager(F2FSynthesis({"Evolution": method}, apply=apply)).run(_circuit())

    assert dict(old.count_ops()) == dict(new.count_ops())


def test_the_filter_is_forwarded():
    with pytest.warns(DeprecationWarning):
        shim = FermionicTrotterization(FermionicSuzukiTrotter(), filter=lambda node: False)

    out = FermionicPassManager(shim).run(_circuit())

    assert dict(out.count_ops()) == {"Evolution": 1}
