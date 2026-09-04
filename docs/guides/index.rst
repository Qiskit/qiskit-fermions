########
Overview
########

This page summarizes the guides that are available in addition to the extensive API documentation of
this package.

.. note::
   These guides might refer to specific Python modules when explaining core concepts.
   Unless otherwise stated, the C API provides matching functionality, although it is not structured
   into modules. However, the function names correspond naturally and should be easy to navigate.

Operators and their structure
=============================

Fermionic operators can be represented in different forms and you can map between these forms as well
as qubit operators using different `mappers`. Start here if you are building a Hamiltonian.

.. toctree::
   :maxdepth: 1
   :glob:

   operators
   grouping
   mappers


Fermionic circuits
==================

You can express circuits directly on the fermionic level rather than mapping to qubits first.

.. toctree::
   :maxdepth: 1
   :glob:

   circuit

Circuit examples
----------------

These guides provide you with concrete examples for different specific use cases:

Chemistry
^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   lucj
   sqdrift

Lattice models
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   skqd


Transpiling fermionic circuits
==============================

Guides explaining how a :class:`.FermionicCircuit` becomes a :class:`~qiskit.circuit.QuantumCircuit`.

.. toctree::
   :maxdepth: 1
   :glob:

   transpilation
   merge_slater_determinant


Workflow examples
=================

.. toctree::
   :maxdepth: 1
   :glob:

   1d_fermi_hubbard
   2d_fermi_hubbard


Workflow validation
===================

.. toctree::
   :maxdepth: 1
   :glob:

   ffsim
