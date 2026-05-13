C Installation Instructions
===========================

Requirements
------------

Before compiling the ``qiskit-fermions`` C API, you need to compile the Qiskit C
API. You can do this in any location, but to get started quickly, follow these
steps:

Compile the Qiskit C API
--------------------------

First, compile the Qiskit C API by following these steps:

.. code:: console

   $ cd path/to/qiskit-fermions
   $ git clone git@github.com:Qiskit/qiskit.git build/qiskit
   $ cd build/qiskit
   $ make c

.. hint::
   On **Windows** only, you will need to copy additional library files:

   .. tab-set::
     .. tab-item:: Windows
        :sync: windows

        .. code:: console

           $ cp target/release/qiskit_cext.dll dist/c/lib/qiskit_cext.dll
           $ cp target/release/qiskit_cext.dll.lib dist/c/lib/qiskit_cext.dll.lib
           $ cp target/release/qiskit_cext.dll.lib dist/c/lib/qiskit.dll.lib

You can now move on to the steps of compiling ``qiskit-fermions`` C API that follow.

If you want to test your installation, install the following
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
      This code uses the ``build/qiskit`` path from the code at
      the top of this page.  Adjust the paths according to your setup.

   .. tab-set::
     .. tab-item:: UNIX
        :sync: unix

        .. code:: console

          $ export QISKIT_LIB=$(find $(pwd)/build/qiskit/dist/c/lib -name "libqiskit.*")
          $ export QISKIT_INCLUDE=$(pwd)/build/qiskit/dist/c/include

     .. tab-item:: Windows
        :sync: windows

        .. code:: console

          $ export QISKIT_LIB=$(find $(pwd)/build/qiskit/dist/c/lib -name "qiskit_cext.dll.lib")
          $ export QISKIT_INCLUDE=$(pwd)/build/qiskit/dist/c/include

3. Compile the ``qiskit-fermions`` C API:

   .. code:: console

      $ make cext

3. (optional) Verify the installation by running the C unit tests:

   .. code:: console

      $ make testc


You should find these relevant files:

- ``dist/c/lib/libqiskit_fermions.so`` (The suffix might vary depending on your operating system)
- ``dist/c/include/qiskit_fermions.h``
