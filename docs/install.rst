Installation Instructions
=========================

.. caution::
   If you are running into issues with the installation, please consult `this
   issue <https://github.com/Qiskit/qiskit-fermions/issues/4>`_. If that does
   not resolve your problem, feel free to comment there with a description of
   your system and problem or leave any general feedback.

Requirements
------------

To compile and install this package, you need to install the following
dependencies:

- a Python (`>=3.10`) virtual environment
- `pip>=25.1`
- `The Rust toolchain <https://rust-lang.org/tools/install/>`_
- `clang <https://clang.llvm.org/>`_

Preparation
-----------

The minimum Qiskit version we require is 2.4.
At the time of writing, this can be installed from a pre-release via PyPI:

.. code:: console

   $ pip install --pre qiskit

If you are interested in compiling the C API of ``qiskit-fermions`` you must
compile Qiskit's C API from source, first.

Please consult the language-specific guide linked below.

.. toctree::
  :hidden:

   C <install-c.rst>
   Python <install-py.rst>
