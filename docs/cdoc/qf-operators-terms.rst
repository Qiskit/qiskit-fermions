.. _qf_operator_terms:

==============
Operator Terms
==============

.. toctree::
   :hidden:
   :maxdepth: 1

   qf-operators-terms-grouping
   qf-operators-terms-filtering

The C API provides various functions to operate on the individual terms of an operator,
partitioning them based on their structure.

Grouping
--------

Please refer to :ref:`grouping_explanation` for a detailed explanation of this module's
functionality.

Members
^^^^^^^

Rather than always relying on the user to provide the group indices themselves,
the C API provides a collection of functions which determine the grouping
information automatically.

.. table::

  ======================================================== ==============================================================
  :c:func:`qf_ferm_op_group_terms_by_electronic_structure` Groups the terms of an operator by their electronic structure.
  ======================================================== ==============================================================

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

.. doxygengroup:: qf_operator_terms
   :content-only:
