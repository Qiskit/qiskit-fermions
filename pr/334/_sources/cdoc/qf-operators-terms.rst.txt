.. _qf_operator_terms:

==============
Operator Terms
==============

.. toctree::
   :hidden:
   :maxdepth: 1

   qf-operators-terms-grouping
   qf-operators-terms-filtering
   qf-operators-terms-ordering

The C API provides various functions to operate on the individual terms of an operator,
partitioning them based on their structure.

Grouping
--------

Refer to :ref:`grouping_explanation` for a detailed explanation of this module's
functionality.

Assignment
^^^^^^^^^^

Rather than always relying on the user to provide the group indices themselves,
the C API provides a collection of functions which determine the grouping
information automatically.

.. table::

  ======================================================== ==============================================================
  :c:func:`qf_ferm_op_group_terms_by_electronic_structure` Groups the terms of an operator by their electronic structure.
  ======================================================== ==============================================================

Analysis
^^^^^^^^

Group indices carry no intrinsic meaning: they only say which terms belong
together, never why. Accordingly, no function reports whether a grouping is
"correct". The C API instead provides a collection of functions that each answer
one narrow, stated question about an existing grouping, so that an assumption a
downstream consumer makes can be checked up front rather than being paid for on
every call. Since these work with generic operator representations, they are
documented on the :doc:`grouping functions <qf-operators-terms-grouping>` page.

Filtering
---------

The C API provides a collection of functions for removing terms from an operator
that do not contribute meaningfully to a downstream computation.

Members
^^^^^^^

.. table::

  ============================================= =======================================================
  :c:func:`qf_ferm_op_filter_diagonal_terms`    Filters out the terms of an operator that are diagonal.
  ============================================= =======================================================

Ordering
--------

The C API provides functions to reorder the terms of an operator into a canonical,
coefficient-independent order.

Members
^^^^^^^

.. table::

  ======================================== ============================================================
  :c:func:`qf_ferm_op_canonical_order`     Reorders the terms of a fermionic operator canonically.
  :c:func:`qf_maj_op_canonical_order`      Reorders the terms of a Majorana operator canonically.
  :c:func:`qf_edge_op_canonical_order`     Reorders the terms of an edge-vertex operator canonically.
  :c:func:`qf_transfer_op_canonical_order` Reorders the terms of a transfer-vertex operator
                                           canonically.
  ======================================== ============================================================

.. doxygengroup:: qf_operator_terms
   :content-only:
