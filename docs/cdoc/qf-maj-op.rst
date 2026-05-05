==================
QfMajoranaOperator
==================

.. c:struct:: QfMajoranaOperator

A Majorana fermion operator.

.. note::
   This is an opaque data structure to the C API whose internals are implemented entirely in Rust.
   The remainder of this page describes the design and related functions to work with this struct.

----

Definition
==========

This operator is defined by a linear combination of products of Majorana operators [1]_, which
can be defined in terms of the standard fermionic second-quantization creation and annihilation
operators (see also :c:struct:`.QfFermionOperator`):

.. math::

    \gamma = a^\dagger + a ~~\text{and}~~ \gamma' = i(a^\dagger - a)

The key property that a Majorana fermion is its own antiparticle becomes immediately apparent:

.. math::

    \gamma_i = \gamma_i^\dagger ~~\text{and}~~ \gamma_i^2 = (\gamma_i^\dagger)^2 = 1

This result in the following anti-commutation relations for :math:`2n` Majorana fermions:

.. math::

    \left\{\gamma_i,\gamma_j\right\} = 2\delta_{ij}

This makes the definition of the entire operator the following:

.. math::

   \text{\texttt{MajoranaOperator}} = \sum_i c_i \bigotimes_j \hat{\gamma_j} \, ,

where :math:`c_i` is the (complex) coefficient making up the linear combination of products of
:math:`\gamma_j`. The index :math:`j` can take any value between 0 and the number of majorana
fermionic modes acted upon by the operator minus 1.

----

.. _qf_maj_op-implementation:

Implementation
==============

This struct stores the terms and coefficients in multiple sparse vectors, akin to the
`compressed sparse row format
<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_
commonly used for sparse matrices. More concretely, a single operator contains 4 arrays:

.. table::

   ============== =================================================================================
   ``coeffs``     A vector of complex coefficients consisting of two 64-bit floating point numbers.
   ``modes``      A vector of 32-bit integers storing the majorana mode indices acted upon.
   ``boundaries`` A vector of integers indicating the boundaries in ``actions`` and ``indices``.
   ============== =================================================================================

The integers in ``modes`` index the Majorana modes, :math:`j`. When using the convenience
function :py:func:`.gamma`, even (odd) indices are used for :math:`\gamma` (:math:`\gamma'`).

.. note::
   You may access **read-only copies** of these internal arrays via their respective functions:

   - :c:func:`qf_maj_op_get_coeffs`
   - :c:func:`qf_maj_op_get_modes`
   - :c:func:`qf_maj_op_get_boundaries`

This data structure allows for very efficient construction and manipulation of operators.
However, it implies that duplicate terms may be contained in an operator at any moment.
These must be resolved manually through the use of :c:func:`qf_maj_op_simplify`.

Construction
------------

A new operator can be constructed directly by specifying the corresponding arrays outlined above.
Alternatively, an empty :c:struct:`QfMajoranaOperator` can be initialized with
:c:func:`qf_maj_op_zero` and terms can be added iteratively via :c:func:`qf_maj_op_add_term`.

.. table::

  ============================  ===================================================
  :c:func:`qf_maj_op_new`       Constructs a new operator from the provided arrays.

  :c:func:`qf_maj_op_zero`      Constructs the additive identity operator.

  :c:func:`qf_maj_op_one`       Constructs the multiplicative identity operator.

  :c:func:`qf_maj_op_add_term`  Adds a term to an existing ``QfMajoranaOperator``.
  ============================  ===================================================

.. note::
   A :c:struct:`QfMajoranaOperator` can be freed with :c:func:`qf_maj_op_free`.

Arithmetics
-----------

The following functions provide arithmetic manipulation:

.. table::

  ===========================  =================================================
  :c:func:`qf_maj_op_add`      Adds two operators together.

  :c:func:`qf_maj_op_mul`      Multiplies an operator by a scalar.

  :c:func:`qf_maj_op_compose`  Composes two operators with each other.

  :c:func:`qf_maj_op_adjoint`  Returns the Hermitian conjugate operator.
  ===========================  =================================================

Manipulation
------------

The following functions provide operator manipulation logic:

.. table::

  ==================================  =========================================================
  :c:func:`qf_maj_op_ichop`           Removes terms with small coefficient magnitudes.
  :c:func:`qf_maj_op_simplify`        Returns an equivalent but simplified operator.
  :c:func:`qf_maj_op_normal_ordered`  Returns an equivalent operator with normal ordered terms.
  :c:func:`qf_maj_op_relabel_modes`   Relabels the modes of an operator.
  ==================================  =========================================================

Properties
----------

The following functions exist to check certain properties of an operator.

.. table::

  ==================================== ===========================================
  :c:func:`qf_maj_op_is_hermitian`     Returns whether an operator is Hermitian.
  :c:func:`qf_maj_op_many_body_order`  Returns the many-body order of an operator.
  :c:func:`qf_maj_op_is_even`          Returns whether an operator is even.
  ==================================== ===========================================

----

.. [1] https://en.wikipedia.org/wiki/Majorana_fermion

Members
=======

.. doxygengroup:: qf_maj_op
   :content-only:
   :members:
   :undoc-members:
