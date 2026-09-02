.. _qf_mapper_library:

==============
Mapper Library
==============

The C API provides efficient implementations of commonly used operator representation mapper
routines.

.. table::

  ============================================ ================================================
  :c:func:`qf_ferm_op_jordan_wigner`           Map a :c:struct:`QfFermionOperator` to a
                                               :external+cqiskit:doc:`QkObs <cdoc/qk-obs>` under
                                               the Jordan-Wigner transformation.
  :c:func:`qf_fermion_to_majorana`             Map a :c:struct:`QfFermionOperator` to a
                                               :c:struct:`QfMajoranaOperator`.
  :c:func:`qf_majorana_to_fermion`             Map a :c:struct:`QfMajoranaOperator` to a
                                               :c:struct:`QfFermionOperator`.
  :c:func:`qf_edge_vertex_to_fermion`          Map a :c:struct:`QfEdgeVertexOperator` to a
                                               :c:struct:`QfFermionOperator`.
  :c:func:`qf_edge_vertex_to_majorana`         Map a :c:struct:`QfEdgeVertexOperator` to a
                                               :c:struct:`QfMajoranaOperator`.
  :c:func:`qf_transfer_vertex_to_fermion`      Map a :c:struct:`QfTransferVertexOperator` to a
                                               :c:struct:`QfFermionOperator`.
  :c:func:`qf_transfer_vertex_to_majorana`     Map a :c:struct:`QfTransferVertexOperator` to a
                                               :c:struct:`QfMajoranaOperator`.
  :c:func:`qf_transfer_vertex_to_edge_vertex`  Map a :c:struct:`QfTransferVertexOperator` to a
                                               :c:struct:`QfEdgeVertexOperator`.
  ============================================ ================================================

----

.. doxygengroup:: qf_mapper_library
   :content-only:
