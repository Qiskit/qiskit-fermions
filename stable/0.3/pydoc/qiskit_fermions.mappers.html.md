<a id="module-qiskit_fermions.mappers"></a>

# Representation Mappers

This module provides a framework for implementing custom representation mapper routines.

## Generator-based mappers

These build a mapper from the Pauli images of an algebra’s generators.

#### NOTE
These four functions do not have a counterpart in the C API.

## Ternary trees

A ternary tree on $N$ nodes defines a fermion-to-qubit encoding of $N$ modes onto
$N$ qubits, and the familiar encodings are particular tree shapes: a chain along `"Z"` is
Jordan-Wigner, a chain along `"X"` is the parity encoding, and the balanced tree attains the
optimal Pauli weight $\lceil \log_3(2N+1) \rceil$. Arbitrary trees (including the irregular
ones grown from a device’s coupling graph) are built by passing a parent-and-label specification to
[`TernaryTree`](../stubs/qiskit_fermions.mappers.TernaryTree.html.md#qiskit_fermions.mappers.TernaryTree) directly, which is what makes this a framework rather than a fixed set of
encodings.

A tree is compiled into a [`TernaryTreeEncoding`](../stubs/qiskit_fermions.mappers.TernaryTreeEncoding.html.md#qiskit_fermions.mappers.TernaryTreeEncoding) once and then passed to any of the
ternary-tree mappers in [`library`](qiskit_fermions.mappers.library.html.md#module-qiskit_fermions.mappers.library). Properties derived from the encoding
itself, such as [`TernaryTreeEncoding.total_parity()`](../stubs/qiskit_fermions.mappers.TernaryTreeEncoding.html.md#qiskit_fermions.mappers.TernaryTreeEncoding.total_parity), are methods on that class.
