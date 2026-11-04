<a id="module-qiskit_fermions.circuit.library.synthesis"></a>

# Circuit Library Synthesis (`F2F`)

This module provides the synthesis methods with which a gate is broken down into a
[`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) of smaller fermionic gates, such as the Trotter formulas for an
[`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution). Pass one to the gate’s `synthesis` argument to choose how it gets decomposed:

```python
Evolution(num_modes, operator, time=1.0, synthesis=FermionicLieTrotter())
```

or choose them for a whole circuit at once with the [`F2FSynthesis`](../stubs/qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis) transpiler pass. Any class
with a `synthesize(gate)` method can leverage a plugin (see [`F2FSynthesisPlugin`](../stubs/qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin)), and the ones
registered under the `qiskit_fermions.circuit.synthesis` entry point are available by name through
the [`F2FSynthesisPluginManager`](../stubs/qiskit_fermions.circuit.library.synthesis.F2FSynthesisPluginManager.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPluginManager).

Both sides of this rewrite live in fermionic space, so these methods can exploit the
[`groups`](../stubs/qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups) of the evolved operator (see
[Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation)) – unlike the fermion-to-qubit synthesis that follows, which only sees the
mapped operator, where the grouping is no longer available.

#### IMPORTANT
This fermion-to-fermion step is **optional**. An [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate can be handed straight to
the fermion-to-qubit stage regardless of how many terms its operator holds; decomposing it in
fermionic space first is a choice, taken to obtain cheaper factors or to expose structure that the
later stage can exploit.

<a id="builtin-f2f-synthesis-plugins"></a>

## Builtin plugins

The table below lists the builtin plugins, organized by the [`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) that they act
upon. These are the entry points of the `qiskit_fermions.circuit.synthesis` group, named
`<gate>.<method>`:

#### An overview of the builtin `qiskit_fermions.circuit.synthesis` entry-points.

| Circuit instruction name                                                                                                           | Plugin method name   | Reference                                                                                                                                                        |
|------------------------------------------------------------------------------------------------------------------------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)                               | `LieTrotter`         | [`FermionicLieTrotter`](../stubs/qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter)                               |
| [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)                               | `SuzukiTrotter`      | [`FermionicSuzukiTrotter`](../stubs/qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter)                         |
| [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)                               | `QDrift`             | [`FermionicQDrift`](../stubs/qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift)                                       |
| [`OrbitalRotation`](../stubs/qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation)                   | `Givens`             | [`FermionicGivensDecomposition`](../stubs/qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition.html.md#qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition)             |
| [`PrepareSlaterDeterminant`](../stubs/qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) | `Givens`             | [`FermionicSlaterGivensDecomposition`](../stubs/qiskit_fermions.circuit.library.synthesis.FermionicSlaterGivensDecomposition.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSlaterGivensDecomposition) |
