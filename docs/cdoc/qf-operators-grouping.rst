.. _qf_operator_grouping:

=================
Operator Grouping
=================

Please refer to :ref:`grouping_explanation` for a detailed explanation of this
functionality.

Library
-------

Rather than always relying on the user to provide the group indices themselves,
the C API provides a collection of functions which determine the grouping
information automatically.

.. table::

  ================================================ ==============================================================
  :c:func:`qf_group_terms_by_electronic_structure` Groups the terms of an operator by their electronic structure.
  ================================================ ==============================================================

----

.. doxygengroup:: qf_operator_grouping
   :content-only:
