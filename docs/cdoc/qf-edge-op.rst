======================
QfEdgeVertexOperator
======================

.. c:struct:: QfEdgeVertexOperator

An edge-vertex operator.

.. note::
   This is an opaque data structure to the C API whose internals are implemented entirely in Rust.
   The remainder of this page describes the design and related functions to work with this struct.

----

.. _qf_edge_op-definition:

Definition
==========

This operator is defined in terms of the edge-vertex (:math:`E_{jk}`, :math:`V_j`) operators:

.. math::

    \begin{align}
    V_j    &= -i \gamma_{2j-1} \gamma_{2j}
            = 1 - 2 a^\dagger_j a_j \, , \nonumber \\
    E_{jk} &= -i \gamma_{2j-1} \gamma_{2k-1}
            = -i (a_j a_k + a_j a^\dagger_k + a^\dagger_j a_k + a^\dagger_j a^\dagger_k)
            = -E_{kj} \nonumber
    \end{align}

which fulfill the following mixed fermionic-bosonic commutation relations for :math:`j \neq k \neq
l \neq m`: [1]_

.. math::

    \begin{align}
    \left\{ E_{jk}, V_k \right\} &= 0 \nonumber \\
    \left\{ E_{jk}, E_{kl} \right\} &= 0 \nonumber \\
    \left[ V_k, V_l \right] &= 0 \nonumber \\
    \left[ E_{jk}, V_l \right] &= 0 \nonumber \\
    \left[ E_{jk}, E_{lm} \right] &= 0 \nonumber \, .
    \end{align}

In summary, edge and vertex operators commute, unless they share *exactly one* index, in which case
they anticommute.

.. note::
   The relations above are stated for :math:`j \neq k \neq l \neq m`, so they do not cover two edge
   operators spanning the *same* pair of modes. Those commute: since :math:`E_{kj} = -E_{jk}`, such
   a pair is collinear, and every operator commutes with itself. This is why the condition is
   "exactly one" shared index rather than "at least one".

We can abuse the notation a little bit and define :math:`V_j = E_{jj}` which reflects how the
internal data structure of this operator works. This makes the definition of the entire operator the
following:

.. math::

   \text{\texttt{EdgeVertexOperator}} = \sum_i c_i \bigotimes_{lr} E_{lr} \, ,

where :math:`lr` indexes the involved operator terms and :math:`c_i` is the (complex) coefficient
making up the linear combination of products. The indices :math:`l` and :math:`r` can take any value
between 0 and the number of fermionic modes acted upon by the operator minus 1.

We will refer to :math:`E_{lr}` as `generalized` edge operators.

----

.. _qf_edge_op-implementation:

Implementation
==============

This struct stores the terms and coefficients in multiple sparse vectors, akin to the
`compressed sparse row format
<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_
commonly used for sparse matrices. More concretely, a single operator contains 4 arrays:

.. table::

   ================= ======================================================================================
   ``coeffs``        A vector of complex coefficients consisting of two 64-bit floating point numbers.
   ``left_indices``  A vector of 32-bit integers storing the `left` fermionic mode indices (:math:`l`).
   ``right_indices`` A vector of 32-bit integers storing the `right` fermionic mode indices (:math:`r`).
   ``boundaries``    A vector of integers indicating the boundaries between terms.
   ================= ======================================================================================

Fermionic modes indexed by ``left_indices`` and ``right_indices`` are considered spinless. The two
index arrays always have the same length, since every generator is identified by exactly one
``(left, right)`` pair; this is why the constructor takes a single ``num_indices`` length for both.

.. note::
   You can access **read-only borrows** of these internal arrays via their respective functions:

   - :c:func:`qf_edge_op_get_coeffs`
   - :c:func:`qf_edge_op_get_left_indices`
   - :c:func:`qf_edge_op_get_right_indices`
   - :c:func:`qf_edge_op_get_boundaries`

   The returned pointers stay valid only until the operator is modified or freed, and must not be
   freed by the caller.

This data structure allows for very efficient construction and manipulation of operators.
However, it implies that duplicate terms might be contained in an operator at any moment.
These must be resolved manually through the use of :c:func:`qf_edge_op_simplify`.

Construction
------------

A new operator can be constructed directly by specifying the corresponding arrays outlined above.
Alternatively, an empty :c:struct:`QfEdgeVertexOperator` can be initialized with
:c:func:`qf_edge_op_zero` and terms can be added iteratively via :c:func:`qf_edge_op_add_term`.

.. table::

  =============================  =====================================================
  :c:func:`qf_edge_op_new`       Constructs a new operator from the provided arrays.

  :c:func:`qf_edge_op_zero`      Constructs the additive identity operator.

  :c:func:`qf_edge_op_one`       Constructs the multiplicative identity operator.

  :c:func:`qf_edge_op_add_term`  Adds a term to an existing ``QfEdgeVertexOperator``.
  =============================  =====================================================

.. note::
   A :c:struct:`QfEdgeVertexOperator` can be freed with :c:func:`qf_edge_op_free`.

Arithmetics
-----------

The following functions provide arithmetic manipulation:

.. table::

  ============================  =================================================
  :c:func:`qf_edge_op_add`      Adds two operators together.

  :c:func:`qf_edge_op_mul`      Multiplies an operator by a scalar.

  :c:func:`qf_edge_op_compose`  Composes two operators with each other.

  :c:func:`qf_edge_op_adjoint`  Returns the Hermitian conjugate operator.
  ============================  =================================================

Manipulation
------------

The following functions provide operator manipulation logic:

.. table::

  ===================================  =========================================================
  :c:func:`qf_edge_op_ichop`           Removes terms with small coefficient magnitudes.
  :c:func:`qf_edge_op_simplify`        Returns an equivalent but simplified operator.
  :c:func:`qf_edge_op_normal_ordered`  Returns an equivalent operator with normal ordered terms.
  :c:func:`qf_edge_op_relabel_modes`   Relabels the modes of an operator.
  ===================================  =========================================================

Properties
----------

The following functions exist to check certain properties of an operator.

.. table::

  ==================================== =======================================================
  :c:func:`qf_edge_op_is_hermitian`    Returns whether an operator is Hermitian.
  :c:func:`qf_edge_op_len`             Returns the number of terms in this operator.
  ==================================== =======================================================

Mapping
-------

The following functions map this operator into another representation:

.. table::

  ========================================== ==============================================
  :c:func:`qf_edge_vertex_to_fermion`        Maps to a :c:struct:`QfFermionOperator`.
  :c:func:`qf_edge_vertex_to_majorana`       Maps to a :c:struct:`QfMajoranaOperator`.
  ========================================== ==============================================

----

.. [1] https://arxiv.org/abs/2512.11418

Members
=======

.. doxygengroup:: qf_edge_op
   :content-only:
   :members:
   :undoc-members:
