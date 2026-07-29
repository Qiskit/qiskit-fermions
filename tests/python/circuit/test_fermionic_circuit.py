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
from qiskit.circuit.library import XGate
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import FermionOperator, ann, cre
from qiskit_fermions.transpiler import FermionicCircuitToDAG


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
    return circ


def test_pickle():
    """Regression test for https://github.com/Qiskit/qiskit-fermions/issues/225.

    A ``FermionicCircuit`` holding an ``Evolution`` gate failed to pickle because the native
    operator it wraps carried no pickle protocol, and separately because its ``__module__``
    was not resolvable via ``importlib`` (see :mod:`qiskit_fermions`'s ``sys.modules`` aliasing).
    """
    circ = _evolution_circuit()
    reconstructed = pickle.loads(pickle.dumps(circ))

    assert reconstructed.modes == circ.modes
    assert reconstructed.count_ops() == circ.count_ops() == {"Evolution": 1}


def test_pickle_dag():
    """Same as :func:`test_pickle`, but for the ``FermionicDAGCircuit`` conversion."""
    dag = FermionicCircuitToDAG().run(_evolution_circuit())
    reconstructed = pickle.loads(pickle.dumps(dag))

    original_nodes = list(dag.op_nodes())
    reconstructed_nodes = list(reconstructed.op_nodes())
    assert len(original_nodes) == len(reconstructed_nodes) == 1

    original_op = original_nodes[0].op.operator
    reconstructed_op = reconstructed_nodes[0].op.operator
    assert original_op.equiv(reconstructed_op)
