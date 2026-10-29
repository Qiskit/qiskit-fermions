<a id="module-qiskit_fermions.circuit.library.synthesis"></a>

# Fermionic Synthesis

This module provides the synthesis methods with which an [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate is broken down into
a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) of smaller fermionic gates. Pass one to the gate’s `synthesis` argument
to choose how its time evolution gets decomposed:

```python
Evolution(num_modes, operator, time=1.0, synthesis=FermionicLieTrotter())
```

Both sides of this rewrite live in fermionic space, so these methods can exploit the
[`groups`](../stubs/qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups) of the evolved operator (see
[Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation)) – unlike the fermion-to-qubit synthesis that follows, which only sees the
mapped operator, where the grouping is no longer available.

#### IMPORTANT
This fermion-to-fermion step is **optional**. An [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate can be handed straight to
the fermion-to-qubit stage regardless of how many terms its operator holds; decomposing it in
fermionic space first is a choice, taken to obtain cheaper factors or to expose structure that the
later stage can exploit.
