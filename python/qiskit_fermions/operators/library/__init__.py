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
================
Operator Library
================

.. currentmodule:: qiskit_fermions.operators.library

Building on top of the operator representations provided by :mod:`~qiskit_fermions.operators`, this
module provides various tools to simplify the construction of commonly used operators.

Commutator Generators
---------------------

Operator classes can implement the :class:`.SupportsCommutators` :py:class:`~typing.Protocol`, which
allows them to be used with the functions listed in the table below, to quickly compute multiple
commutator variants.

.. autosummary::
   :toctree: ../stubs/

   SupportsCommutators
   commutator
   anti_commutator
   double_commutator

Common Operators
----------------

Various common operators can easily be generated from constructor methods.
This section provides an overview of these methods for a quick reference, grouped by category.

Electronic Integrals
^^^^^^^^^^^^^^^^^^^^

The constructor methods listed here generally take the coefficients of electronic structure
Hamiltonians as an input. Different flavors exist:

* ``tril``: these methods consume 1-dimensional arrays of flattened (generalized) triangular indices
* ``full``: these methods consume high-dimensional arrays
* ``spin``: these methods take separate arrays for the different spin species
* ``sym``: these methods take a single array for one spin species and infer the other spin species

* 1-Body Terms

.. table::

   ================================================= ===========================================================
   :meth:`.FermionOperator.from_1body_tril_spin_sym` Constructs from spin-symmetric triangular 1-body integrals.
   :meth:`.FermionOperator.from_1body_tril_spin`     Constructs from separate spin triangular 1-body integrals.
   ================================================= ===========================================================

* 2-Body Terms

.. table::

   ================================================= ===========================================================
   :meth:`.FermionOperator.from_2body_tril_spin_sym` Constructs from spin-symmetric triangular 2-body integrals.
   :meth:`.FermionOperator.from_2body_tril_spin`     Constructs from separate spin triangular 2-body integrals.
   ================================================= ===========================================================

Other Generators
----------------

Finally, the following additional operator generator utilities exist in this module:

.. autosummary::
   :toctree: ../stubs/

   FCIDump
"""

from qiskit_fermions._lib.operators.operators_library.fcidump import FCIDump

from .commutators import (
    SupportsCommutators,
    anti_commutator,
    commutator,
    double_commutator,
)

__all__ = [
    "FCIDump",
    "SupportsCommutators",
    "anti_commutator",
    "commutator",
    "double_commutator",
]
