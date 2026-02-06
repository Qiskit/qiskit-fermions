Installation Instructions
=========================

.. caution::
   The installation process of this package has not been ironed out yet.
   Thus, for now it is important that you follow these instructions carefully.

Requirements
------------

To compile and install this package, you need to install the following
dependencies:

- a Python (`>=3.10`) virtual environment
- `The Rust toolchain <https://rust-lang.org/tools/install/>`_
- `clang <https://clang.llvm.org/>`_

Preparation
-----------

Currently, the build process assumes a locally compiled version of Qiskit to be
located at ``build/qiskit``. Thus, the following steps are required:

1. activate your Python environment

2. (only the first time) clone this repository:

   .. code:: console

      $ git clone git@github.com:Qiskit/qiskit-fermions.git

3. change into the correct directory:

   .. code:: console

      $ cd qiskit-fermions

4. (only the first time) clone Qiskit into the desired location:

   .. code:: console

      $ git clone git@github.com:Qiskit/qiskit.git build/qiskit

5. compile Qiskit locally (first the C API, then the Python package):

   .. code:: console

      $ cd build/qiskit
      $ make c
      $ pip install -r requirements.txt -c constraints.txt
      $ SETUPTOOLS_RUST_CARGO_PROFILE=release pip install -e .

6. symlink the compiled library (replace ``*`` to match your local filename and
   update the ``.so`` suffix if it differs on your platform):

   .. code:: console

      $ cd qiskit/
      $ ln -s _accelerate.*.so libqiskit.so

7. verify that everything worked by running the ``qiskit-fermions`` Rust tests:

   .. code:: console

      $ cd ../../..
      $ make testrust

Assuming everything above worked, you can now proceed with installing the C and
Python APIs for this package!

.. toctree::
  :hidden:

   C <install-c.rst>
   Python <install-py.rst>
