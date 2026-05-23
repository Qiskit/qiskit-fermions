###############
Qiskit Fermions
###############

.. warning::
   This package is under active development!
   If you have feedback, please `open an issue <https://github.com/Qiskit/qiskit-fermions/issues/new/choose>`_.

This package extends Qiskit with tools for working on fermionic systems.
It follows a similar design philosophy as Qiskit:

- The core functionality is written in Rust.
- First-party language bindings are provided for both Python and C.
- The APIs are designed to "feel" similar to Qiskit.

The scope of this package includes the following functionality:

- Several operator data structures
- A framework to develop operator conversion methods
- A library of common operator converters
- A framework to develop quantum circuit synthesis methods
- A library of common quantum circuit synthesizers

Installation
------------

Refer to the `installation instructions <install.rst>`_.


Deprecation policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
`release notes <https://qiskit.github.io/qiskit-fermions/release-notes.html>`_.


Contributing
------------

Follow the instructions in the `Developer guide <https://github.com/Qiskit/qiskit-fermions/blob/main/CONTRIBUTING.md>`_.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-fermions/issues/new/choose>`_ for tracking requests and bugs.


License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-fermions/blob/main/LICENSE.txt>`_
