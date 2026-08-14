.. _qf_operator_library:

================
Operator Library
================

.. toctree::
   :hidden:
   :maxdepth: 1

   qf-commutators
   qf-electronic-integrals
   qf-fcidump

The C API provides various functions to simplify the construction of commonly used operators.

Commutator Generators
---------------------

.. table::

  ====================================== ==================================================
  :c:func:`qf_op_type_commutator`        Computes the commutator of two operators.
  :c:func:`qf_op_type_anti_commutator`   Computes the anti-commutator of two operators.
  :c:func:`qf_op_type_double_commutator` Computes the double-commutator of three operators.
  ====================================== ==================================================

Common Operators
----------------

This section provides a quick reference for various operator constructor functions, grouped by
category.

Electronic Integrals
^^^^^^^^^^^^^^^^^^^^

.. table::

  ============================================= =====================================================
  :c:func:`qf_ferm_op_from_1body_tril_spin_sym` Constructs from spin-symmetric triangular 1-body
                                                integrals.
  :c:func:`qf_ferm_op_from_1body_tril_spin`     Constructs from separate spin-species triangular
                                                1-body integrals.
  :c:func:`qf_ferm_op_from_2body_tril_spin_sym` Constructs from spin-symmetric triangular 2-body
                                                integrals.
  :c:func:`qf_ferm_op_from_2body_tril_spin`     Constructs from separate spin-species triangular
                                                2-body integrals.
  ============================================= =====================================================

Other Generators
----------------

Finally, the following additional operator generator utilities exist:

.. table::

  ===================== ======================================================
  :c:struct:`QfFCIDump` An electronic structure Hamiltonian in FCIDump format.
  ===================== ======================================================


.. doxygengroup:: qf_operator_library
   :content-only:
