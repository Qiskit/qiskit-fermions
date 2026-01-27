Python Installation Instructions
================================

Assuming you have already completed the `base installation Instructions
<install.rst>`_, the rest of the Python installation instructions is fairly
simple:

1. change into the correct directory:

   .. code:: console

      $ cd path/to/qiskit-fermions

2. install the Python installation tooling

   .. code:: console

      $ pip install setuptools setuptools_rust

3. compile the `pyext` crate:

   .. code:: console

      $ make pyext

4. install the Python package:

   .. code:: console

      $ make pyinstall

5. get the correct ``LD_LIBRARY_PATH``:

   .. code:: console

      $ make echo_ld_lib_path

6. ensure that you copy and run the output above

7. verify that everything worked by running the Python API tests:

   .. code:: console

      $ make testpython


From now on, you need to ensure that you always export the ``LD_LIBRARY_PATH``
as before when you want to use this Python package.
