Python Installation Instructions
================================

.. warning::
   Windows is not yet fully supported by the Python API of ``qiskit-fermions``.
   We are actively working on resolving this. Stay tuned for updates!

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
   ``--no-build-isolation`` to ensure that ``qiskit`` is available correctly:

   .. code:: console

      $ pip install --no-build-isolation .

   .. hint::

      You can also perform an editable install while still compiling the
      underlying Rust crate in ``release`` mode like so:

      .. code:: console

         $ SETUPTOOLS_RUST_CARGO_PROFILE=release pip install --no-build-isolation -e .

5. (optional) Verify the installation by running the Python unittests:

   .. code:: console

      $ pip install --group test
      $ make testpython
