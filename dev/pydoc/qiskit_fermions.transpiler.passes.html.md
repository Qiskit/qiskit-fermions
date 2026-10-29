<a id="module-qiskit_fermions.transpiler.passes"></a>

# Transpiler Passes

This module provides various transpiler passes for the stages explained in
[`qiskit_fermions.transpiler`](qiskit_fermions.transpiler.html.md#module-qiskit_fermions.transpiler).

<a id="qiskit-fermions-transpiler-passes-optimization"></a>

## Optimization Passes

These passes provide different kinds of optimization of [`FermionicDAGCircuit`](../stubs/qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit) instances.

<a id="qiskit-fermions-transpiler-passes-layout"></a>

## Layouting Passes

These passes are designed to help configuring the global
[`F2QLayout`](../stubs/qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout) setting for the transpilation process. This setting
needs to be placed in the `f2q_layout` field of the
[`property_set`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState#property_set), from where it will be read during the
[Synthesis Passes](#qiskit-fermions-transpiler-passes-synthesis).

<a id="qiskit-fermions-transpiler-passes-synthesis"></a>

## Synthesis Passes

The main logic for mapping a [`FermionicDAGCircuit`](../stubs/qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit) to a
[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit) is implemented by a single synthesis pass, namely the
[`F2QSynthesis`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis). It is conceptually similar to Qiskit’s
[`HighLevelSynthesis`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.HighLevelSynthesis) because it simply iterates over the
[`FermionicDAGCircuit`](../stubs/qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit) instructions and delegates the mapping to qubit-based instructions to
[Transpiler Pass Synthesis (F2Q)](qiskit_fermions.transpiler.passes.synthesis.html.md#qiskit-fermions-transpiler-passes-synthesis-plugins) for each of the encountered types of
[`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate).
