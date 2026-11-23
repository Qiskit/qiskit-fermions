########
Overview
########

.. meta::
   :description: Overview of the Qiskit Fermions guides, grouped by topic, with links to each one.

..
    _This page may *NOT* use any `.. toctree::` directives! Instead, the guides'
    toctree lives in the `docs/guides/index.rst` page. Here, all guides should
    be categorized into a section using a :ref: reference.

This page summarizes the guides that are available in addition to the extensive API documentation of
this package.

.. note::
   These guides might refer to specific Python modules when explaining core concepts.
   Unless otherwise stated, the C API provides matching functionality, although it is not structured
   into modules. However, the function names correspond naturally and should be easy to navigate.

To get started with the breadth of features this package supports, check out
these end-to-end guides:

- :ref:`1d_fermi_hubbard`
- :ref:`2d_fermi_hubbard`


The following sections outline additional guides that dive into more specific
topics in detail.


Operators and their structure
=============================

Fermionic operators can be represented in different forms and you can map between these forms as well
as qubit operators using different `mappers`. Start here if you are building a Hamiltonian.

- :ref:`operators_explanation`
- :ref:`grouping_explanation`
- :ref:`mappers_explanation`


Fermionic circuits
==================

You can express circuits directly on the fermionic level rather than mapping to qubits first.

- :ref:`fermionic_circuit_explanation`

Circuit examples
----------------

These guides provide you with concrete examples for different specific use cases:

Chemistry
^^^^^^^^^

- :ref:`sqdrift_getting_started`

Lattice models
^^^^^^^^^^^^^^

- :ref:`skqd_getting_started`


Transpiling fermionic circuits
==============================

Guides explaining how a :class:`.FermionicCircuit` becomes a :class:`~qiskit.circuit.QuantumCircuit`.

- :ref:`transpilation_explanation`
- :ref:`fermionic_synthesis_explanation`
- :ref:`merge_slater_determinant_explanation`
- :ref:`lucj_transpilation`


Relationship to other packages
==============================

How ``qiskit-fermions`` divides the work with the other tools of the ecosystem.

- :ref:`ffsim_relationship_explanation`
