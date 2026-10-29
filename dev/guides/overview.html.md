# Overview

This page summarizes the guides that are available in addition to the extensive API documentation of
this package.

#### NOTE
These guides might refer to specific Python modules when explaining core concepts.
Unless otherwise stated, the C API provides matching functionality, although it is not structured
into modules. However, the function names correspond naturally and should be easy to navigate.

To get started with the breadth of features this package supports, check out
these end-to-end guides:

- [Simulate 1D Fermi-Hubbard dynamics with flow sets](1d_fermi_hubbard.html.md#d-fermi-hubbard)
- [Simulate 2D Fermi-Hubbard dynamics with flow sets](2d_fermi_hubbard.html.md#d-fermi-hubbard)

The following sections outline additional guides that dive into more specific
topics in detail.

## Operators and their structure

Fermionic operators can be represented in different forms and you can map between these forms as well
as qubit operators using different mappers. Start here if you are building a Hamiltonian.

- [Design principles of operator representations](operators.html.md#operators-explanation)
- [Group operator terms: use the operator structure](grouping.html.md#grouping-explanation)
- [Map operator representations](mappers.html.md#mappers-explanation)

## Fermionic circuits

You can express circuits directly on the fermionic level rather than mapping to qubits first.

- [Work with fermionic circuits](circuit.html.md#fermionic-circuit-explanation)

### Circuit examples

These guides provide you with concrete examples for different specific use cases:

#### Chemistry

- [Generate SqDRIFT circuits](sqdrift.html.md#sqdrift-getting-started)

#### Lattice models

- [Generate Krylov time-evolution circuits (SKQD)](skqd.html.md#skqd-getting-started)

## Transpiling fermionic circuits

Guides explaining how a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) becomes a [`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit).

- [Transpile fermionic circuits](transpilation.html.md#transpilation-explanation)
- [Synthesize an Evolution gate in fermionic space](fermionic_synthesis.html.md#fermionic-synthesis-explanation)
- [Optimize Slater determinant preparation](merge_slater_determinant.html.md#merge-slater-determinant-explanation)
- [Transpile an LUCJ ansatz](lucj.html.md#lucj-transpilation)

## Relationship to other packages

How `qiskit-fermions` divides the work with the other tools of the ecosystem.

- [How this package relates to ffsim](ffsim.html.md#ffsim-relationship-explanation)
