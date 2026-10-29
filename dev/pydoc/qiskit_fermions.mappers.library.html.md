<a id="module-qiskit_fermions.mappers.library"></a>

# Mapper Library

This module provides efficient implementations of commonly used operator representation mapper
routines.

## Python wrappers

The functions below are thin, type-agnostic convenience mappers. Each delegates to the appropriate
library implementation below under the hood, based on the type of the object it is given. They do
not implement any conversion logic themselves.

<a id="jordan-wigner-mappers"></a>

## Library implementations

The functions below are the native, efficient mapper implementations backing this module.

<a id="ternary-tree-mappers"></a>

## Ternary-tree encodings

These map an operator through a ternary-tree encoding, which is built with
[`TernaryTree`](../stubs/qiskit_fermions.mappers.TernaryTree.html.md#qiskit_fermions.mappers.TernaryTree) and
[`TernaryTreeEncoding`](../stubs/qiskit_fermions.mappers.TernaryTreeEncoding.html.md#qiskit_fermions.mappers.TernaryTreeEncoding) from the
[`mappers`](qiskit_fermions.mappers.html.md#module-qiskit_fermions.mappers) framework module. Compile a tree once and reuse the encoding across
every operator mapped through it. The type-agnostic [`ternary_tree()`](../stubs/qiskit_fermions.mappers.library.ternary_tree.html.md#qiskit_fermions.mappers.library.ternary_tree) wrapper dispatches to these.
