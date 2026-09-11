Install the C API
=================

Prerequisites
^^^^^^^^^^^^^

Before getting started, you must ensure that you have installed the `Rust
toolchain <https://www.rust-lang.org/tools/install>`_ as well as the `clang
compiler <https://clang.llvm.org/>`_.

Now, clone the ``qiskit-fermions`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-fermions.git
    cd qiskit-fermions

Next, you must also compile the Qiskit C API. You can do this in any location,
but to get started quickly, follow these steps:

.. code:: sh

    git clone git@github.com:Qiskit/qiskit.git build/qiskit
    cd build/qiskit
    ## git-checkout the latest stable tag
    make c

.. hint::
   On **Windows** only, you need to copy additional library files:

   .. tab-set::
     .. tab-item:: Windows
        :sync: windows

        .. code:: sh

            cp target/release/qiskit_cext.dll dist/c/lib/qiskit_cext.dll
            cp target/release/qiskit_cext.dll.lib dist/c/lib/qiskit_cext.dll.lib
            cp target/release/qiskit_cext.dll.lib dist/c/lib/qiskit.dll.lib

Now you can follow the steps in the next section to compile the ``qiskit-fermions`` C API.

Compile from source
^^^^^^^^^^^^^^^^^^^

To get started, configure your shell environment with the locations of your
compiled Qiskit C API library and include directory.

.. important::
   The C API links against the Qiskit **C** library, so ``QISKIT_LIB`` and
   ``QISKIT_INCLUDE`` must resolve to it. If unset, the build attempts to infer
   them from any installed ``qiskit`` Python package which bundles its C
   artifacts (``qiskit.capi.get_lib()``/``get_include()``); set them explicitly,
   as below, to build against a separately compiled Qiskit C API such as the
   ``build/qiskit`` clone above.

.. hint::
   This code uses the ``build/qiskit`` path from the code at
   the top of this page. Adjust the paths according to your setup.

.. tab-set::
   .. tab-item:: UNIX
      :sync: unix

      .. code:: sh

          export QISKIT_LIB=$(find $(pwd)/build/qiskit/dist/c/lib -name "libqiskit.*")
          export QISKIT_INCLUDE=$(pwd)/build/qiskit/dist/c/include

   .. tab-item:: Windows
      :sync: windows

      .. code:: sh

          export QISKIT_LIB=$(find $(pwd)/build/qiskit/dist/c/lib -name "qiskit_cext.dll.lib")
          export QISKIT_INCLUDE=$(pwd)/build/qiskit/dist/c/include

Now, compile the ``qiskit-fermions`` C API:

.. code:: sh

    make cext

If you want to test your installation, you now must also install `CMake
<https://cmake.org/>`_.
You can then test your compilation by running the C unit tests:

.. code:: sh

    make testc


Finally, verify that you see these relevant files:

- ``dist/c/lib/libqiskit_fermions.so`` (The suffix might vary depending on your operating system)
- ``dist/c/include/qiskit_fermions.h``
