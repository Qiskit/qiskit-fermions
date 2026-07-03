Install the Python API
=======================

Requirements
------------

Qiskit must be installed in the same Python environment that you
install ``qiskit-fermions`` in. The simplest way to do this is from
PyPI, but you can install Qiskit from source if you want to.

.. code:: console

   $ pip install qiskit

Steps
-----

Assuming that you have already installed Qiskit into your Python environment
(see `here <install.rst>`_), the remaining installation process is fairly
simple:

1. Ensure that you are in the right directory:

   .. code:: console

      $ cd path/to/qiskit-fermions

2. Install the Python installation tooling:

   .. code:: console

      $ pip install --group build

3. Install the ``qiskit-fermions`` Python package into your environment with
   ``--no-build-isolation`` to ensure that ``qiskit`` is available:

   .. code:: console

      $ pip install --no-build-isolation .

   .. hint::

      You can also perform an editable install while still compiling the
      underlying Rust crate in ``release`` mode:

      .. code:: console

         $ SETUPTOOLS_RUST_CARGO_PROFILE=release pip install --no-build-isolation -e .

4. (optional) You may wish to generate the Python type stub files to have better
   integration with coding tools (e.g. for tab completion). You can do that
   quite simply like so:

   .. code:: console

      $ make pystubs

5. (optional) Verify that the installation was successful:
   The simplest test is to try and import one of the classes provided by the
   ``qiskit-fermions`` package, for example like so:

   .. code:: console

      $ python -c "from qiskit_fermions.circuit import FermionicCircuit"

   If this completes successfully, your installation worked.

   .. hint::

      If you have performed an editable install, you can also run the entire
      Python test suite:

      .. code:: console

         $ pip install --group test
         $ make testpython
