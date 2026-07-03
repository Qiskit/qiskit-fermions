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

# ruff: noqa: D205,D212,D415
"""
==========
Transpiler
==========

.. currentmodule:: qiskit_fermions.transpiler

--------
Overview
--------

Transpilation is the process of rewriting a given input circuit to conform with desired criteria
such as topological and operational constraints of the hardware used to execute the final circuit.
We are not going to explain this in more detail here, and instead refer to Qiskit's documentation of
the :mod:`qiskit.transpiler` module.

The focus here lies on explaining how we achieve the transpilation of a :class:`.FermionicCircuit`
to a :class:`~qiskit.circuit.QuantumCircuit`.

------
Stages
------

Conceptually, we split the transpilation process into several stages:

==================================================== ==========================================
Stage                                                Description
==================================================== ==========================================
:ref:`qiskit_fermions-transpiler-stage-input`        convert to DAG data structure
:ref:`qiskit_fermions-transpiler-stage-optimization` fermionic-level optimization
:ref:`qiskit_fermions-transpiler-stage-layout`       fermion-to-qubit layouting
:ref:`qiskit_fermions-transpiler-stage-synthesis`    fermion-to-qubit synthesis
:ref:`qiskit_fermions-transpiler-stage-qubit`        continued transpilation on the qubit-level
:ref:`qiskit_fermions-transpiler-stage-output`       convert from DAG data structure
==================================================== ==========================================

.. _qiskit_fermions-transpiler-stage-input:

Input
^^^^^

The various transpiler passes are implemented to work with :class:`.FermionicDAGCircuit` as the
underlying data structure to store the circuit operations. Such `directed acyclic graphs` (DAGs)
provide an efficient data model for the traversal and manipulation of circuits.

However, an end-user is more likely to work with a :class:`.FermionicCircuit` as it provides a more
intuitive interface and data model. As such, this input stage simply runs the
:class:`.FermionicCircuitToDAG` transpiler pass.

.. _qiskit_fermions-transpiler-stage-optimization:

Optimization
^^^^^^^^^^^^

This stage of the transpilation pipeline can implement circuit optimizations while preserving the
type of circuit to be an instance of :class:`.FermionicDAGCircuit`. As such, no qubit information is
required (or necessarily available) at this point in the transpilation pipeline.

.. _qiskit_fermions-transpiler-stage-layout:

Layout
^^^^^^

One global configuration setting for the transpilation process is the fermion-to-qubit `"layout"`.
This must be provided by the user and it must match the provided fermion-to-qubit mapping (in the
sense that, if a chosen mapping encodes a fixed number of fermions with a different number of
qubits, the configured layout must account for that).

In the general case, fermionic modes are not always encoded with an occupation-basis into the qubit
register. Consequently, we cannot associate a single fermionic mode with a single qubit.
Therefore, the user-provided fermion-to-qubit layout (:type:`F2QLayout`) associates
:type:`~qiskit_fermions.circuit.FermionicRegister` instances with
:class:`~qiskit.circuit.QuantumRegister` ones.

.. autosummary::
   :toctree: ../stubs/

   F2QLayout

The way to configure this global setting, is by placing a :type:`F2QLayout` instance in the
``f2q_layout`` field of the :attr:`~qiskit.transpiler.PassManager.property_set`. For more details
refer to :ref:`qiskit_fermions-transpiler-passes-layout`.

.. _qiskit_fermions-transpiler-stage-synthesis:

Synthesis
^^^^^^^^^

At its core, the transpilation is handled by the :class:`.F2QSynthesis` transpiler pass. It is
conceptually similar to Qiskit's :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass, which
uses various `plugins` for transpiling high-level circuit instructions. For more details, refer to
the documentation of :class:`.F2QSynthesis` directly.

How a given :class:`.FermionicGate` can be synthesized in terms of qubit-based operations will
depend on the particular gate type as well as the user-chosen fermion-to-qubit mapping. For more
details, refer to :ref:`qiskit_fermions-transpiler-passes-synthesis-plugins`.

.. _qiskit_fermions-transpiler-stage-qubit:

Qubit
^^^^^

At this point in the transpilation process, we have reached a
:class:`~qiskit.dagcircuit.DAGCircuit` instance and can continue to use Qiskit's transpilation
pipeline as one would usually.

.. hint::
   Additional transpiler passes for optimizations on the qubit-level that take into account the
   knowledge of a circuit originating from a :class:`.FermionicCircuit` may be added in the future!

.. _qiskit_fermions-transpiler-stage-output:

Output
^^^^^^

This stage implements effectively the reverse of :ref:`qiskit_fermions-transpiler-stage-input`, by
calling the :class:`.QuantumDAGToCircuit` transpiler pass.

-------------
Pass Managers
-------------

All of the stages above are the default stages by how the :mod:`~qiskit_fermions.transpiler.presets`
orchestrate the various :class:`~qiskit.passmanager.BasePassManager`s in their resulting
:class:`~qiskit.passmanager.MultiStagePassManager`.

The individual stages can be either a single pass (when they change from one internal representation
(IR) to another, such as the :ref:`qiskit_fermions-transpiler-stage-input`,
:ref:`qiskit_fermions-transpiler-stage-synthesis`, and
:ref:`qiskit_fermions-transpiler-stage-output` stages above) or a stage can be a
:class:`~qiskit.passmanager.BasePassManager` whose internal passes can be modified.

For the stages operating on fermionic circuits (:ref:`qiskit_fermions-transpiler-stage-optimization`
and :ref:`qiskit_fermions-transpiler-stage-layout`), the type of passmanager to use is
:class:`.FermionicPassManager`.

.. autosummary::
   :toctree: ../stubs/

   FermionicPassManager

Conversion Passes
^^^^^^^^^^^^^^^^^

Some very basic conversion passes are provided directly by this module:

.. autosummary::
   :toctree: ../stubs/

   FermionicCircuitToDAG
   FermionicDAGToCircuit
   QuantumDAGToCircuit

Presets
^^^^^^^

For user convenience, the :mod:`qiskit_fermions.transpiler.presets` module provides a number of
functions for quickly building pre-defined transpilation pipelines.
"""

from __future__ import annotations

from typing import TypeAlias

from qiskit.circuit import QuantumRegister
from qiskit.passmanager import GenericPass

from qiskit_fermions.circuit import FermionicDAGCircuit, FermionicRegister

from .converters import FermionicCircuitToDAG, FermionicDAGToCircuit, QuantumDAGToCircuit
from .passmanager import FermionicPassManager

FermionicDAGCircuitPass: TypeAlias = GenericPass[FermionicDAGCircuit, FermionicDAGCircuit]
"""The type definition of a fermionic-to-fermionic generic transpiler pass."""

F2QLayout: TypeAlias = dict[FermionicRegister, QuantumRegister]
"""A mapping of fermionic mode registers to quantum registers.

Users must provide a data structure of this type to the circuit transpiler. In a trivial case, (such
as under the Jordan-Wigner transformation (cf. :func:`.jordan_wigner`)) every fermionic mode gets
mapped to a single qubit. However, not all fermion-to-qubit mappings are of this occupation-based
nature, which is why we do not associate single fermionic modes with single qubits. Crucially, the
size of the registers on either side of this mapping may differ.
"""

__all__ = [
    "F2QLayout",
    "FermionicCircuitToDAG",
    "FermionicDAGToCircuit",
    "FermionicPassManager",
    "QuantumDAGToCircuit",
]
