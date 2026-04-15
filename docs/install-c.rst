C Installation Instructions
===========================

Requirements
------------

Before compiling the C API of ``qiskit-fermions`` you need to compile Qiskit's C
API. You may do this in any location but to get your started quickly, these
steps can get you started quickly:

.. code:: console

   $ cd path/to/qiskit-fermions
   $ git clone git@github.com:Qiskit/qiskit.git build/qiskit
   $ cd build/qiskit
   $ make c

At this point, you should have successfully compiled Qiskit's C API. You can now
move on to the steps of compiling ``qiskit-fermions`` C API below.

If you want to test your installation, you will need to install the following
dependencies:

- `CMake <https://cmake.org/>`_

Steps
-----

1. Ensure that you are in the right directory:

   .. code:: console

      $ cd path/to/qiskit-fermions

2. Configure your shell environment with the locations of your compiled Qiskit C
   API library and include directory.

   .. hint::
      Here we assume the ``build/qiskit`` path from the quickstart example at
      the top, you need to adjust the paths according to your setup.

   .. code:: console

      $ export QISKIT_LIB=$(find $(pwd)/build/qiskit/dist/c/lib -name "libqiskit.*")
      $ export QISKIT_INCLUDE=$(pwd)/build/qiskit/dist/c/include

3. Compile the C API of ``qiskit-fermions``:

   .. code:: console

      $ make cext

3. (optional) Verify the installation by running the C unittests:

   .. code:: console

      $ make testc


You should now find these relevant files:

- ``dist/c/lib/libqiskit_fermions.so`` (the suffix can vary depending on your OS)
- ``dist/c/include/qiskit_fermions.h``
