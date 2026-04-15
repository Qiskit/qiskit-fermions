Python Installation Instructions
================================

Assuming that you have already installed Qiskit into your Python environment
(see `here <install.rst>`_), the remaining installation process is fairly
simple:

1. Ensure that you are in the right directory:

   .. code:: console

      $ cd path/to/qiskit-fermions

2. Install the Python installation tooling:

   .. code:: console

      $ pip install --group build

3. Configure your shell environment with the location of your Qiskit
   installation.

   .. hint::
      You _may_ be able to skip this step, depending on how ``pip`` resolves the
      location of the installed Qiskit during the build process. If you _do_ try
      to skip this step and obtain an error that the ``_accelerate.*.so``
      library cannot be found, ensure that you set these environment variables
      before trying the next step again.

   .. code:: console

      $ export QISKIT_LIB=$(python -c "import qiskit; print(qiskit.capi.get_lib())")
      $ export QISKIT_INCLUDE=$(python -c "import qiskit; print(qiskit.capi.get_include())")

4. Install the ``qiskit-fermions`` Python package into your environment:

   .. code:: console

      $ pip install .

   .. hint::

      You can also perform an editable install while still compiling the
      underlying Rust crate in ``release`` mode like so:

      .. code:: console

         $ SETUPTOOLS_RUST_CARGO_PROFILE=release pip install -e .

5. (optional) Verify the installation by running the Python unittests:

   .. code:: console

      $ pip install --group test
      $ make testpython
