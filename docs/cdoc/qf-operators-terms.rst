.. _qf_operator_terms:

==============
Operator Terms
==============

.. toctree::
   :hidden:
   :maxdepth: 1

   qf-operators-terms-grouping

The C API provides various functions to operator on the individual terms of an operator,
partitioning them based on their structure.

Grouping
--------

Please refer to :ref:`grouping_explanation` for a detailed explanation of this module's
functionality.

Library
^^^^^^^

Rather than always relying on the user to provide the group indices themselves,
the C API provides a collection of functions which determine the grouping
information automatically.

.. table::

  ================================================ ==============================================================
  :c:func:`qf_group_terms_by_electronic_structure` Groups the terms of an operator by their electronic structure.
  ================================================ ==============================================================

.. doxygengroup:: qf_operator_terms
   :content-only:
