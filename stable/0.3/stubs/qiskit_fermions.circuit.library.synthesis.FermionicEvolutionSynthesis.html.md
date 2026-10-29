# FermionicEvolutionSynthesis

### *class* FermionicEvolutionSynthesis

The interface for fermion-to-fermion synthesis of an [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate.

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

An implementation of this interface rewrites the time evolution $e^{-i t H}$ carried by an
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate into a [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) of *smaller* [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)
instances. Both sides of that rewrite live in fermionic space, which is what distinguishes this
from the fermion-to-qubit synthesis performed later by [`F2QSynthesis`](qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis): the fermionic
structure of `H` (in particular its [`groups`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups)
(see [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation))) survives and is still available to that later stage.

#### IMPORTANT
This fermion-to-fermion step is **optional**. An [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate does not need to be
broken down in fermionic space at all: the fermion-to-qubit stage can map and synthesize it
directly, however many terms its operator holds. Splitting it up first is a *choice*, made
because the resulting factors are individually cheaper to implement, or because the split
exposes structure (such as mutually commuting groups) that the later stage can exploit.

Because the fermion-to-fermion and fermion-to-qubit steps compose, an evolution may end up
approximated at either level, at both, or at neither. Where an approximation is involved at both,
the accuracy of the result is governed by the weaker of the two.

#### NOTE
Qiskit’s [`EvolutionSynthesis`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.synthesis.EvolutionSynthesis) is the closest analogue, but it belongs
to the qubit layer: it synthesizes a [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate) into a
[`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit), after the operator has already been mapped.

### Methods

#### *abstractmethod* synthesize(gate)

Synthesizes the provided [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate.

* **Parameters:**
  **gate** ([*Evolution*](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)) – the gate to synthesize. Its
  [`operator`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.operator) is the Hermitian operator
  $H$ and its `params[0]` the evolution time $t$ of $e^{-i t H}$.
* **Returns:**
  A [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) on `gate.num_modes` modes implementing the evolution.
* **Return type:**
  [FermionicCircuit](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)
