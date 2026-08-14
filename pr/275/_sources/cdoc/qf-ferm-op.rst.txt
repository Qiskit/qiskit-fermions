=================
QfFermionOperator
=================

.. c:struct:: QfFermionOperator

A spin-less fermionic operator.

.. note::
   This is an opaque data structure to the C API whose internals are implemented entirely in Rust.
   The remainder of this page describes the design and related functions to work with this struct.

----

Definition
==========

This operator is defined by a linear combination of products of fermionic creation and
annihilation operators acting on spin-less fermionic modes. That is to say, the individual
terms fulfill the following anti-commutation relations: [1]_

.. math::

    \left\{a^\dagger_i, a^\dagger_j\right\} =
    \left\{a_i, a_j\right\} = 0,~~\text{and}~~
    \left\{a_i, a^\dagger_j\right\} = \delta_{ij} \, ,

where :math:`i` and :math:`j` do not distinguish the spin species of the fermionic
modes they are indexing.

This makes the definition of the entire operator the following:

.. math::

   \text{\texttt{QfFermionOperator}} = \sum_i c_i \bigotimes_j \hat{A_j} \, ,

where :math:`\hat{A_j} \in \{ a_j, a^\dagger_j \}` and :math:`c_i` is the (complex) coefficient
making up the linear combination of products. The index :math:`j` can take any value between 0
and the number of fermionic modes acted upon by the operator minus 1.

----

.. _qf_ferm_op-implementation:

Implementation
==============

This struct stores the terms and coefficients in multiple sparse vectors, akin to the
`compressed sparse row format
<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_
commonly used for sparse matrices. More concretely, a single operator contains 4 arrays:

.. table::

   ============== =================================================================================
   ``coeffs``     A vector of complex coefficients consisting of two 64-bit floating point numbers.
   ``actions``    A vector of booleans storing the nature of the second-quantization actions.
   ``modes``      A vector of 32-bit integers storing the fermionic mode indices acted upon.
   ``boundaries`` A vector of integers indicating the boundaries in ``actions`` and ``modes``.
   ============== =================================================================================

Entries in ``actions`` indicate creation (annihilation) operators by ``True`` (``False``).
Fermionic modes indexed by ``modes`` are considered spinless.

.. note::
   You may access **read-only copies** of these internal arrays via their respective functions:

   - :c:func:`qf_ferm_op_get_coeffs`
   - :c:func:`qf_ferm_op_get_actions`
   - :c:func:`qf_ferm_op_get_modes`
   - :c:func:`qf_ferm_op_get_boundaries`

This data structure allows for very efficient construction and manipulation of operators.
However, it implies that duplicate terms may be contained in an operator at any moment.
These must be resolved manually through the use of :c:func:`qf_ferm_op_simplify`.

Construction
------------

A new operator can be constructed directly by specifying the corresponding arrays outlined above.
Alternatively, an empty :c:struct:`QfFermionOperator` can be initialized with
:c:func:`qf_ferm_op_zero` and terms can be added iteratively via :c:func:`qf_ferm_op_add_term`.

.. table::

  =============================  ===================================================
  :c:func:`qf_ferm_op_new`       Constructs a new operator from the provided arrays.

  :c:func:`qf_ferm_op_zero`      Constructs the additive identity operator.

  :c:func:`qf_ferm_op_one`       Constructs the multiplicative identity operator.

  :c:func:`qf_ferm_op_add_term`  Adds a term to an existing ``QfFermionOperator``.
  =============================  ===================================================

.. note::
   A :c:struct:`QfFermionOperator` can be freed with :c:func:`qf_ferm_op_free`.

Arithmetics
-----------

The following functions provide arithmetic manipulation:

.. table::

  ============================  =================================================
  :c:func:`qf_ferm_op_add`      Adds two operators together.

  :c:func:`qf_ferm_op_mul`      Multiplies an operator by a scalar.

  :c:func:`qf_ferm_op_compose`  Composes two operators with each other.

  :c:func:`qf_ferm_op_adjoint`  Returns the Hermitian conjugate operator.
  ============================  =================================================

Manipulation
------------

The following functions provide operator manipulation logic:

.. table::

  ====================================  =========================================================
  :c:func:`qf_ferm_op_ichop`            Removes terms with small coefficient magnitudes.
  :c:func:`qf_ferm_op_simplify`         Returns an equivalent but simplified operator.
  :c:func:`qf_ferm_op_normal_ordered`   Returns an equivalent operator with normal ordered terms.
  :c:func:`qf_ferm_op_relabel_modes`    Relabels the modes of an operator.
  ====================================  =========================================================

Properties
----------

The following functions exist to check certain properties of an operator.

.. table::

  ==============================================  ==========================================================
  :c:func:`qf_ferm_op_is_hermitian`               Returns whether an operator is Hermitian.
  :c:func:`qf_ferm_op_max_rank`                   Returns the maximum rank of the terms in this operator.
  :c:func:`qf_ferm_op_conserves_particle_number`  Returns whether an operator is particle-number conserving.
  ==============================================  ==========================================================

----

.. [1] https://en.wikipedia.org/wiki/Second_quantization#Fermion_creation_and_annihilation_operators

Members
=======

.. doxygengroup:: qf_ferm_op
   :content-only:
   :members:
   :undoc-members:
