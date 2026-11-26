<a id="module-qiskit_fermions.mappers.optimization"></a>

# Mapper Optimizations

This module provides various routines for optimizing fermion-to-qubit mappings.
Their specific nature can vary greatly, so refer to the docstrings of each individual function for
more details.

## Ternary-tree search

A ternary tree defines a fermion-to-qubit encoding (see [`TernaryTree`](../stubs/qiskit_fermions.mappers.TernaryTree.html.md#qiskit_fermions.mappers.TernaryTree)), and which tree you
pick decides the encoding’s cost. The routine below chooses one from information the balanced tree
ignores.
