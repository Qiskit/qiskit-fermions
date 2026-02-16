"""Optional dependency testers.

This module defines lazy availability checks for optional third-party
dependencies. The checkers are instances of :class:`qiskit.utils.LazyImportTester`
and can be used as Booleans (evaluated lazily), or to raise a
:class:`qiskit.exceptions.MissingOptionalLibraryError` when a dependency is
required.

Available Testers
=================

.. py:data:: HAS_PYOMO

   `Pyomo <https://www.pyomo.org/>`__ is a Python-based optimization modeling
   language used to build linear and mixed-integer programs (LP/MILP) for
   classical solvers.
"""

from qiskit.utils import LazyImportTester

HAS_PYOMO = LazyImportTester("pyomo", name="Pyomo", install="pip install pyomo")
