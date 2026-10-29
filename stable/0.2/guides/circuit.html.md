<a id="fermionic-circuit-explanation"></a>

# Work with fermionic circuits

#### IMPORTANT
The concepts in this guide are currently available only in the Python API.
Equivalent functionality will be made available in the C API in a future
release.

This guide explains how to use the [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) to implement
quantum algorithms directly in fermionic space, rather than mapping to qubits
first.

## Why use fermionic circuits

Consider the traditional workflow for implementing the time evolution of a
fermionic Hamiltonian on a qubit-based architecture:

1. Define the fermionic Hamiltonian (for example, $H = \sum_{pq} h_{pq}
   c^\dagger_p c_q$).
2. Map it to a qubit operator (for example, by using the Jordan-Wigner mapping).
3. Implement the time evolution in qubit space (using Trotterization or other
   methods).

This approach has a fundamental limitation; early mapping to qubits discards
problem-aware knowledge about the fermionic structure. Standard optimization
passes cannot exploit fermionic symmetries or allowed term reorderings,
resulting in suboptimal circuits with unnecessary depths and gate counts.

The [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) enables a better workflow:

1. Define the fermionic Hamiltonian (for example, $H = \sum_{pq} h_{pq}
   c^\dagger_p c_q$).
2. Implement the time evolution directly in fermionic space using fermionic
   gates.
3. Map to qubit space as part of the transpilation process.

This approach addresses the limitation by keeping problem-aware knowledge in
fermionic space, where optimization passes can exploit fermionic structure and
commutation relations to reduce circuit depth. The fermionic circuit describes
what computation to perform, while transpilation handles how to map it to
qubits, providing a cleaner separation of concerns.

## Generic mode indexing

Both the [`operators`](../pydoc/qiskit_fermions.operators.html.md#module-qiskit_fermions.operators) module and the
[`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) use generic mode-based indexing that makes no
assumptions about the nature of the modes. A mode is simply an abstract index
labeling a fermionic degree of freedom, with no inherent semantics. This design
choice mirrors the operators module and ensures maximum flexibility.
For example, you can implement time evolution for any operator implementing the
[`OperatorTrait`](../stubs/qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait) protocol, regardless of its mathematical representation.

## Build a fermionic circuit

The example below implements the time evolution described in the previous section.
It constructs a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) by specifying the number of fermionic
modes, then adds fermionic gates from the [`qiskit_fermions.circuit.library`](../pydoc/qiskit_fermions.circuit.library.html.md#module-qiskit_fermions.circuit.library)
to implement the time evolution.

The example also demonstrates how to incorporate domain knowledge into an
operator by using [operator term grouping](grouping.html.md#grouping-explanation). By
assigning group indices to the Hamiltonian terms, you preserve structural
information that optimization passes can exploit throughout the transpilation
stack.

### Python

```pycon
>>> from qiskit_fermions.circuit import FermionicCircuit
>>> from qiskit_fermions.circuit.library import Evolution
>>> from qiskit_fermions.operators import FermionOperator, cre, ann
>>>
>>> # Create a circuit with 4 fermionic modes
>>> circuit = FermionicCircuit(4)
>>>
>>> # Define a simple fermionic Hamiltonian
>>> hamiltonian = FermionOperator.from_terms([
...     ([cre(0), ann(2)], 0.5),
...     ([cre(2), ann(0)], 0.5),
...     ([cre(1), ann(3)], 0.5),
...     ([cre(3), ann(1)], 0.5),
... ])
>>> # Add some problem structure by grouping our Hamiltonian terms
>>> hamiltonian.groups = [0, 0, 1, 1]
>>>
>>> # Add an evolution gate to implement exp(-i * t * H)
>>> evolution = Evolution(4, hamiltonian, time=1.0)
>>> circuit.append(evolution, circuit.register)
```

### C

```c
// The C API for FermionicCircuit is not available yet.
```

([`png`](_downloads/c656c99107bb439d5d056fe13666448b/circuit-2.png), [`hires.png`](_downloads/c978e8c5fa01e74032ace32415615702/circuit-2.hires.png), [`pdf`](_downloads/68981126cc731a750e988d02093d416c/circuit-2.pdf))

![A simple `FermionicCircuit` with a single time evolution gate.](../_images/circuit-2.png)

([`png`](_downloads/79564271ec7e02f69d7225eb17a40b1e/circuit-3.png), [`hires.png`](_downloads/a87b0986e371500adf3e1dd3a51c49e8/circuit-3.hires.png), [`pdf`](_downloads/7f17e3c00ad4fc5c51df7fc7d21c0393/circuit-3.pdf))

![A decomposed `FermionicCircuit` with several time evolution gates.](../_images/circuit-3.png)

Notice how the operator term grouping is preserved even in simple operations
like decomposition. This demonstrates how structural information flows through the circuit
stack. To understand the full transpilation to
qubits, refer to the [Transpiling fermionic circuits](transpilation.html.md#transpilation-explanation) guide.

## Transpile fermionic circuits

To implement the quantum algorithm represented by your
[`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) it must be transpiled to a
[`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit). You can learn how to do this in the
[Transpile fermionic circuits](transpilation.html.md#transpilation-explanation) guide.

## Next steps

- Learn about available fermionic gates in the [`qiskit_fermions.circuit.library`](../pydoc/qiskit_fermions.circuit.library.html.md#module-qiskit_fermions.circuit.library)
  documentation.
- Explore the [operators explanation guide](operators.html.md#operators-explanation) to understand
  how to construct fermionic Hamiltonians that you can use with fermionic circuits.
- Review the [SQDRIFT](sqdrift.html.md#sqdrift-getting-started) getting started guide for a practical
  end-to-end example.
- Read the [ffsim relationship guide](ffsim.html.md#ffsim-relationship-explanation) to understand how a
  [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) can be simulated directly, and how these generic mode indices take on
  spin semantics only at simulation time.
