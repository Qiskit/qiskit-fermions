C Installation Instructions
===========================

Requirements
------------

First, you must prepare the installation by following the `base installation
Instructions <install.rst>`_.

If you want to test your installation, you need to install the following
dependencies:

- `CMake <https://cmake.org/>`_

Steps
-----

Now, the rest of the C installation instructions are fairly simple:

1. change into the correct directory:

   .. code:: console

      $ cd path/to/qiskit-fermions

2. compile the `cext` crate:

   .. code:: console

      $ make cext

3. verify that everything worked by running the C API tests:

   .. code:: console

      $ make testc


You should now find these relevant files:

- `dist/c/lib/libqiskit_fermions.so` (the suffix can vary depending on your OS)
- `dist/c/include/qiskit_fermions.h`
