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

The focus here lies on explaining how we achieve the transpilation of a :class:`.FermionicCircuit` to
a :class:`~qiskit.circuit.QuantumCircuit`.

------
Stages
------

Conceptually, we split the transpilation process into several stages:

==================================================== ==========================================
Stage                                                Description
==================================================== ==========================================
:ref:`qiskit_fermions-transpiler-stage-optimization` fermionic-level optimization
:ref:`qiskit_fermions-transpiler-stage-layout`       fermion-to-qubit layouting
:ref:`qiskit_fermions-transpiler-stage-synthesis`    fermion-to-qubit synthesis
:ref:`qiskit_fermions-transpiler-stage-quantum`      continued transpilation on the qubit-level
==================================================== ==========================================

.. _qiskit_fermions-transpiler-stage-optimization:

Optimization
^^^^^^^^^^^^

This stage of the transpilation pipeline can implement circuit optimizations while preserving the
type of circuit to be an instance of :class:`.FermionicCircuit`. As such, no qubit information is
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

How a given :class:`.FermionicGate` can be synthesized in terms of qubit-based operations will depend
on the particular gate type as well as the user-chosen fermion-to-qubit mapping. For more details,
refer to :ref:`qiskit_fermions-transpiler-passes-synthesis-plugins`.

.. _qiskit_fermions-transpiler-stage-quantum:

Quantum
^^^^^^^

At this point in the transpilation process, we have reached :class:`~qiskit.circuit.QuantumCircuit`
instance and can continue to use Qiskit's transpilation pipeline as one would usually.

.. hint::
   Additional transpiler passes for optimizations on the qubit-level that take into account the
   knowledge of a circuit originating from a :class:`.FermionicCircuit` may be added in the future!

-------------
Pass Managers
-------------

Qiskit's transpilation process is orchestrated by a :class:`~qiskit.transpiler.PassManager`.
In particular, a :class:`~qiskit.transpiler.StagedPassManager` can be used to orchestrate the
transpilation into stages as explained above.

Here, we are dealing with a change of circuit representation converting :class:`.FermionicCircuit`
instances to :external:class:`~qiskit.circuit.QuantumCircuit` ones. As such, this module provides
its own interfaces of these transpiler pass managers listed below.

.. note::
   Qiskit is currently working on native support of transpiler pipelines involving more than a
   single intermediate representation. Once that gets more formalized, the implementation here
   will be aligned with the resulting interfaces. See also `this tracking issue
   <https://github.com/Qiskit/qiskit/issues/16115>`_.

.. autosummary::
   :toctree: ../stubs/

   FermionicPassManager
   FermionicStagedPassManager
   FermionicToQubitConverter

Presets
^^^^^^^

For user convenience, the :mod:`qiskit_fermions.transpiler.presets` module provides a number of
functions for quickly building pre-defined transpilation pipelines.
"""

from __future__ import annotations

from typing import TypeAlias

from qiskit.circuit import QuantumRegister

from qiskit_fermions.circuit import FermionicRegister

from .passmanager import FermionicPassManager, FermionicStagedPassManager, FermionicToQubitConverter

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
    "FermionicPassManager",
    "FermionicStagedPassManager",
    "FermionicToQubitConverter",
]
