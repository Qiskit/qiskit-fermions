==========================
QfTransferVertexOperator
==========================

.. c:struct:: QfTransferVertexOperator

A transfer-vertex operator.

.. note::
   This is an opaque data structure to the C API whose internals are implemented entirely in Rust.
   The remainder of this page describes the design and related functions to work with this struct.

----

.. _qf_transfer_op-definition:

Definition
==========

This operator is defined in terms of the transfer-vertex (:math:`T_{jk}`, :math:`V_j`) operators:

.. math::

    \begin{align}
    V_j    &= -i \gamma_{2j-1} \gamma_{2j}
            = 1 - 2 a^\dagger_j a_j \, , \nonumber \\
    T_{jk} &= \frac{i}{2} V_j E_{jk}
            = \frac{i}{2} \gamma_{2j} \gamma_{2k-1} \, , \nonumber \\
    T_{kj} &= \frac{i}{2} E_{jk} V_k
            = -\frac{i}{2} \gamma_{2j-1} \gamma_{2k} \nonumber
    \end{align}

where :math:`E_{jk}` is an edge operator of the :c:struct:`QfEdgeVertexOperator` and these
individual terms fulfill the following mixed fermionic-bosonic commutation relations for
:math:`j \lt k \lt l \lt m`: [1]_

.. math::

    \begin{align}
    \left\{ T_{jk}, V_k \right\} &= 0 \nonumber \\
    \left\{ T_{jk}, T_{lk} \right\} &= 0 \nonumber \\
    \left[ V_k, V_l \right] &= 0 \nonumber \\
    \left[ T_{jk}, V_l \right] &= 0 \nonumber \\
    \left[ T_{jk}, T_{lm} \right] &= 0 \nonumber \\
    \left[ T_{jk}, T_{kj} \right] &= 0 \nonumber \\
    \left[ T_{jk}, T_{km} \right] &= 0 \nonumber \, .
    \end{align}

.. caution::
   Unlike the edge operators, where :math:`E_{kj} = -E_{jk}` makes the two orientations two
   representations of a single operator, :math:`T_{jk}` and :math:`T_{kj}` are *different*
   operators. The order of the two index arrays is therefore significant, and there is no
   orientation convention to normalize — which is why
   :c:func:`qf_transfer_op_normal_ordered` takes no ``ascending`` parameter where
   :c:func:`qf_edge_op_normal_ordered` does.

We can abuse the notation a little bit and define :math:`V_j = T_{jj}` which reflects how the
internal data structure of this operator works. This makes the definition of the entire operator the
following:

.. math::

   \text{\texttt{TransferVertexOperator}} = \sum_i c_i \bigotimes_{lr} T_{lr} \, ,

where :math:`lr` indexes the involved operator terms and :math:`c_i` is the (complex) coefficient
making up the linear combination of products. The indices :math:`l` and :math:`r` can take any value
between 0 and the number of fermionic modes acted upon by the operator minus 1.

We will refer to :math:`T_{lr}` as `generalized` transfer operators.

----

.. _qf_transfer_op-implementation:

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

   - :c:func:`qf_transfer_op_get_coeffs`
   - :c:func:`qf_transfer_op_get_left_indices`
   - :c:func:`qf_transfer_op_get_right_indices`
   - :c:func:`qf_transfer_op_get_boundaries`

   The returned pointers stay valid only until the operator is modified or freed, and must not be
   freed by the caller.

This data structure allows for very efficient construction and manipulation of operators.
However, it implies that duplicate terms might be contained in an operator at any moment.
These must be resolved manually through the use of :c:func:`qf_transfer_op_simplify`.

Construction
------------

A new operator can be constructed directly by specifying the corresponding arrays outlined above.
Alternatively, an empty :c:struct:`QfTransferVertexOperator` can be initialized with
:c:func:`qf_transfer_op_zero` and terms can be added iteratively via
:c:func:`qf_transfer_op_add_term`.

.. table::

  =================================  =========================================================
  :c:func:`qf_transfer_op_new`       Constructs a new operator from the provided arrays.

  :c:func:`qf_transfer_op_zero`      Constructs the additive identity operator.

  :c:func:`qf_transfer_op_one`       Constructs the multiplicative identity operator.

  :c:func:`qf_transfer_op_add_term`  Adds a term to an existing ``QfTransferVertexOperator``.
  =================================  =========================================================

.. note::
   A :c:struct:`QfTransferVertexOperator` can be freed with :c:func:`qf_transfer_op_free`.

Arithmetics
-----------

The following functions provide arithmetic manipulation:

.. table::

  ================================  =================================================
  :c:func:`qf_transfer_op_add`      Adds two operators together.

  :c:func:`qf_transfer_op_mul`      Multiplies an operator by a scalar.

  :c:func:`qf_transfer_op_compose`  Composes two operators with each other.

  :c:func:`qf_transfer_op_adjoint`  Returns the Hermitian conjugate operator.
  ================================  =================================================

Manipulation
------------

The following functions provide operator manipulation logic:

.. table::

  =======================================  =========================================================
  :c:func:`qf_transfer_op_ichop`           Removes terms with small coefficient magnitudes.
  :c:func:`qf_transfer_op_simplify`        Returns an equivalent but simplified operator.
  :c:func:`qf_transfer_op_normal_ordered`  Returns an equivalent operator with normal ordered terms.
  :c:func:`qf_transfer_op_relabel_modes`   Relabels the modes of an operator.
  =======================================  =========================================================

Properties
----------

The following functions exist to check certain properties of an operator.

.. table::

  ======================================== =======================================================
  :c:func:`qf_transfer_op_is_hermitian`    Returns whether an operator is Hermitian.
  :c:func:`qf_transfer_op_len`             Returns the number of terms in this operator.
  ======================================== =======================================================

Mapping
-------

The following functions map this operator into another representation:

.. table::

  ============================================== ==================================================
  :c:func:`qf_transfer_vertex_to_fermion`        Maps to a :c:struct:`QfFermionOperator`.
  :c:func:`qf_transfer_vertex_to_majorana`       Maps to a :c:struct:`QfMajoranaOperator`.
  :c:func:`qf_transfer_vertex_to_edge_vertex`    Maps to a :c:struct:`QfEdgeVertexOperator`.
  ============================================== ==================================================

----

.. [1] https://arxiv.org/abs/2512.11418

Members
=======

.. doxygengroup:: qf_transfer_op
   :content-only:
   :members:
   :undoc-members:
