# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# ruff: noqa: D205,D212,D415
"""
=====================
Optional Dependencies
=====================

.. currentmodule:: qiskit_fermions.utils.optionals

This module defines lazy availability testers for optional third-party dependencies. The checkers
are instances of :external:class:`~qiskit.utils.LazyImportTester` and can be used as booleans
(evaluated lazily), or to raise a :external:class:`~qiskit.exceptions.MissingOptionalLibraryError`
when a dependency is required.

-----------------
Available Testers
-----------------

.. autodata:: HAS_FFSIM
.. autodata:: HAS_PYOMO
.. autodata:: HAS_QISKIT_ADDON_SQD
"""

from qiskit.utils import LazyImportTester

HAS_FFSIM = LazyImportTester("ffsim", name="ffsim", install='pip install "qiskit-fermions[ffsim]"')
"""`ffsim <https://github.com/qiskit-community/ffsim>`__ is a high-performance simulator for
fermionic quantum circuits that exploits particle-number and spin-Z conservation.

:class:`~qiskit_fermions.circuit.FermionicCircuit` instances can be simulated using
:external:func:`ffsim.apply_unitary`.

.. note::
   ``ffsim`` does not support Windows, which is why it is an optional dependency.

.. seealso::
   :external:class:`~qiskit.utils.LazyDependencyManager` for usage examples and the available
   methods of this object.
"""

HAS_PYOMO = LazyImportTester(
    "pyomo", name="Pyomo", install='pip install "qiskit_fermions[optimization]"'
)
"""`Pyomo <https://www.pyomo.org/>`__ is a Python-based optimization modeling language used to build
linear and mixed-integer programs (LP/MILP) for classical solvers.

.. seealso::
   :external:class:`~qiskit.utils.LazyDependencyManager` for usage examples and the available
   methods of this object.
"""

HAS_QISKIT_ADDON_SQD = LazyImportTester(
    "qiskit_addon_sqd", name="qiskit-addon-sqd", install='pip install "qiskit-addon-sqd"'
)
"""`qiskit-addon-sqd <https://github.com/Qiskit/qiskit-addon-sqd>`__ implements sample-based quantum
diagonalization (SQD): it projects a Hamiltonian onto the subspace spanned by measured computational-
basis configurations and iteratively refines that subspace via *configuration recovery*.

.. note::
   ``qiskit-addon-sqd`` pulls in PySCF, which does not support Windows, which is why it is an optional
   dependency.

.. seealso::
   :external:class:`~qiskit.utils.LazyDependencyManager` for usage examples and the available
   methods of this object.
"""
