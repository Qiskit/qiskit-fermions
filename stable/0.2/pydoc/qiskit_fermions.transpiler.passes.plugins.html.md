<a id="qiskit-fermions-transpiler-passes-synthesis-plugins"></a>

# Transpiler Pass Plugins

The [`F2QSynthesis`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis) transpiler pass exposes a plugin interface (see
[`F2QSynthesisPlugin`](../stubs/qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin)) through which implementations for mapping fermion-based instructions
to qubit-based ones can be registered.

The available plugins are managed by the [`F2QSynthesisPluginManager`](../stubs/qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager). Choosing the right
combination of synthesis plugins is crucial and the [`F2QSynthesis`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis) transpilation pass does
**not** have any default plugins defined! For this, refer to the transpiler
[`presets`](qiskit_fermions.transpiler.presets.html.md#module-qiskit_fermions.transpiler.presets).

The plugins to use at runtime have to be registered in the [`F2QSynthesis.methods`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis.methods) attribute.
This can be done easily through the `config` argument of [`F2QSynthesis`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis) which specifies
the plugins to use from the [`F2QSynthesisPluginManager`](../stubs/qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager) or it can be done manually by
writing into [`F2QSynthesis.methods`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis.methods) directly.

The table below lists the builtin synthesis plugins, organized by the [`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) that
they act upon:

<a id="builtin-synthesis-plugins"></a>

#### An overview of the builtin `qiskit_fermions.transpiler.synthesis` entry-points.

| Circuit instruction name                                                                                                           | Plugin method name          | Reference                                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)                               | `MapperFn`                  | [`MapperFnEvolutionSynthesis`](../stubs/qiskit_fermions.transpiler.passes.synthesis.MapperFnEvolutionSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.MapperFnEvolutionSynthesis)                                       |
| [`InitializeModes`](../stubs/qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes)                   | `TrivialOccupation`         | [`TrivialOccupationInitializeModesSynthesis`](../stubs/qiskit_fermions.transpiler.passes.synthesis.TrivialOccupationInitializeModesSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.TrivialOccupationInitializeModesSynthesis)         |
| [`OrbitalRotation`](../stubs/qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation)                   | `GivensDecomposition`       | [`GivensDecompositionOrbitalRotationSynthesis`](../stubs/qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionOrbitalRotationSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionOrbitalRotationSynthesis)     |
| [`PrepareSlaterDeterminant`](../stubs/qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) | `GivensDecompositionSlater` | [`GivensDecompositionSlaterDeterminantSynthesis`](../stubs/qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.GivensDecompositionSlaterDeterminantSynthesis) |

## Additional helper functions

The Pauli term order that a mapper function produces is preserved through synthesis, since a product
formula synthesizes the terms in the order it receives them. These adapters wrap a mapper function to
choose that order:
