###############
Qiskit Fermions
###############

``qiskit-fermions`` extends the Qiskit SDK with tools for working on fermionic
systems. Within its scope it provides the following:

- Efficient data structures for the representation and manipulation of fermionic
  operators in different forms
- A framework for implementing operator conversion methods (including
  fermion-to-qubit encodings)
- A library of efficient implementations of common conversion methods
- A framework and library of gates for expressing fermionic circuits
- A transpilation pipeline integrated with the Qiskit transpiler process to
  synthesize fermionic circuits into qubit-based circuits

Additionally, ``qiskit-fermions`` integrates with other tools of the ecosystem,
such as:

- `ffsim <https://qiskit-community.github.io/ffsim/>`_ for efficient simulation
  of fermionic circuits

Get started
-----------

Several guides exist to help you get started with this package. For an overview of its breadth of features, visit the
:doc:`1D Fermi-Hubbard guide <guides/1d_fermi_hubbard>`.

Use case examples
-----------------

Components of this package have been used in research related to the following
papers:

- The qDRIFT randomized circuit compilation in [1]_.
- The fermion-to-qubit synthesis during the transpilation process in [2]_.

Technical discussion
--------------------

Design intentions
"""""""""""""""""

This package is deliberately designed to align with Qiskit: it builds on a core
implemented in Rust and provides first-party language bindings to Python and C.
Its API intends to draw parallels to Qiskit in order to seamlessly integrate
into the workflows of users with experience in programming Qiskit.

A core principle to the design of ``qiskit-fermions`` was the decoupling of its
fermionic circuit representation from its qubitized form. To be more precise:
the fermionic circuits are meaningful by themselves and do not require a mapping
to qubit space to be interpretable.
Furthermore, the fermionic circuit representation cannot make any assumptions
about its fermion-to-qubit encoding applied later on. Consequently, while
Jordan-Wigner retains a dominant position and role, it is not assumed to be the
_default_ fermion-to-qubit encoding.

Known issues
""""""""""""

As long as the Qiskit C API has not yet reached feature parity with its Python
API, some components of this package remain exclusive to its Python API, too.
This includes the entire circuit library (:mod:`qiskit_fermions.circuit`) as
well as transpiler passes (:mod:`qiskit_fermions.transpiler`).

Future work
"""""""""""

- Migrate the circuit library and transpiler passes into the Rust core (and
  provide a C API for interacting with them)
- Extend the library of efficient operator conversion implementations
- Extend the library of efficient operator data structures

Contributing
------------

The source code is available `on GitHub
<https://github.com/Qiskit/qiskit-fermions>`_.

The developer guide is located at `CONTRIBUTING.md
<https://github.com/Qiskit/qiskit-fermions/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct
<https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

Citing this package
-------------------

If you use this package in your research, use the `CITATION.bib
<https://github.com/Qiskit/qiskit-fermions/blob/main/CITATION.bib>`_ file in
this project's repository to cite the appropriate reference(s).

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-fermions/blob/main/LICENSE.txt>`_

Deprecation policy
------------------

This package follows `semantic versioning <https://semver.org/>`_. Breaking changes
are made only occasionally, to improve the user experience. When possible, old
interfaces are kept and marked as deprecated for as long as they can co-exist
with the new ones. Each substantial improvement, breaking change, or deprecation
is documented in the `release notes
<https://quantum.cloud.ibm.com/docs/api/qiskit-fermions/release-notes>`_.

References
----------

.. [1] Samuele Piccinelli, et al., `Quantum chemistry with provable convergence
   via randomized sample-based Krylov quantum diagonalization
   <https://arxiv.org/abs/2508.02578v2>`_, arXiv:2508.02578 [quant-ph].

.. [2] Anthony Gandon, et al., `Stabilizer-based quantum simulation of fermion
   dynamics with local qubit encodings <https://arxiv.org/abs/2512.11418v2>`_,
   arXiv:2512.11418 [quant-ph].


.. toctree::
  :hidden:

   Documentation Home <self>
   Installation Instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/qiskit-fermions>

.. toctree::
   :hidden:
   :caption: Tutorials

   Simulate 1D Fermi-Hubbard dynamics with flow sets <guides/1d_fermi_hubbard>
   Simulate 2D Fermi-Hubbard dynamics with flow sets <guides/2d_fermi_hubbard>

.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <https://quantum.cloud.ibm.com/docs/api/qiskit-fermions>
   C API reference <https://quantum.cloud.ibm.com/docs/api/qiskit-fermions-c>
   Release notes <release-notes>
