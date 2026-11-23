.. _qf_operators:

************************
Operator Representations
************************

.. toctree::
   :hidden:
   :maxdepth: 1

   qf-ferm-op
   qf-maj-op
   qf-edge-op
   qf-transfer-op

The C API provides various data structures for representing fermionic operators in different bases.

.. table::

  ==================================== ============================================================
  :c:struct:`QfFermionOperator`        An operator in the basis of the second-quantization creation
                                       and annihilation operators.
  :c:struct:`QfMajoranaOperator`       An operator in the basis of Majorana fermions.
  :c:struct:`QfEdgeVertexOperator`     An operator in the basis of the edge and vertex operators.
  :c:struct:`QfTransferVertexOperator` An operator in the basis of the transfer and vertex
                                       operators.
  ==================================== ============================================================

Shared helpers
==============

A few functions are not specific to any one representation and so are provided once for all of them.

.. table::

  ============================ ===================================================================
  :c:func:`qf_edges_free`      Frees an edge array returned by a non-commutation-edges accessor.
  ============================ ===================================================================

.. doxygengroup:: qf_operators
   :content-only:
   :members:
   :undoc-members:
