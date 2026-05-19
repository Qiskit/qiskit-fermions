Install ``qiskit-fermions``
===========================

.. hint::
   Support for Windows is still a work-in-progress. Compiling the C API should
   be possible, but the Python API is still not fully functional. We are working
   on improving this situation. In the meantime, we suggest that you use the
   Linux subsystem on Windows.

Requirements
------------

To compile and install this package, you need to install the following
dependencies:

- A Python (`>=3.10`) virtual environment
- `pip>=25.1`
- `The Rust toolchain <https://rust-lang.org/tools/install/>`_
- `clang <https://clang.llvm.org/>`_

Installation
------------

Consult the appropriate guide for the language bindings you intend to use:

.. toctree::
  :hidden:

   C <install-c.rst>
   Python <install-py.rst>
