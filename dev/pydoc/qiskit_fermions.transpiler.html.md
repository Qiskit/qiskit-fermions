<a id="module-qiskit_fermions.transpiler"></a>

# Transpiler

## Overview

Transpilation is the process of rewriting a given input circuit to conform with desired criteria
such as topological and operational constraints of the hardware used to execute the final circuit.
We are not going to explain this in more detail here, and instead refer to Qiskit’s documentation of
the [`qiskit.transpiler`](https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler#module-qiskit.transpiler) module.

The focus here lies on explaining how we achieve the transpilation of a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)
to a [`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit).

## Stages

Conceptually, we split the transpilation process into several stages:

| Stage                                                       | Description                                |
|-------------------------------------------------------------|--------------------------------------------|
| [Input](#qiskit-fermions-transpiler-stage-input)        | convert to DAG data structure              |
| [Optimization](#qiskit-fermions-transpiler-stage-optimization) | fermionic-level optimization               |
| [Layout](#qiskit-fermions-transpiler-stage-layout)       | fermion-to-qubit layouting                 |
| [Synthesis](#qiskit-fermions-transpiler-stage-synthesis)    | fermion-to-qubit synthesis                 |
| [Qubit](#qiskit-fermions-transpiler-stage-qubit)        | continued transpilation on the qubit-level |
| [Output](#qiskit-fermions-transpiler-stage-output)       | convert from DAG data structure            |

<a id="qiskit-fermions-transpiler-stage-input"></a>

### Input

The various transpiler passes are implemented to work with [`FermionicDAGCircuit`](../stubs/qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit) as the
underlying data structure to store the circuit operations. Such directed acyclic graphs (DAGs)
provide an efficient data model for the traversal and manipulation of circuits.

However, an end-user is more likely to work with a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) as it provides a more
intuitive interface and data model. As such, this input stage simply runs the
[`FermionicCircuitToDAG`](../stubs/qiskit_fermions.transpiler.FermionicCircuitToDAG.html.md#qiskit_fermions.transpiler.FermionicCircuitToDAG) transpiler pass.

<a id="qiskit-fermions-transpiler-stage-optimization"></a>

### Optimization

This stage of the transpilation pipeline can implement circuit optimizations while preserving the
type of circuit to be an instance of [`FermionicDAGCircuit`](../stubs/qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit). As such, no qubit information is
required (or necessarily available) at this point in the transpilation pipeline.

<a id="qiskit-fermions-transpiler-stage-layout"></a>

### Layout

One global configuration setting for the transpilation process is the fermion-to-qubit “layout”.
This must be provided by the user and it must match the provided fermion-to-qubit mapping (in the
sense that, if a chosen mapping encodes a fixed number of fermions with a different number of
qubits, the configured layout must account for that).

In the general case, fermionic modes are not always encoded with an occupation-basis into the qubit
register. Consequently, we cannot associate a single fermionic mode with a single qubit.
Therefore, the user-provided fermion-to-qubit layout ([`F2QLayout`](../stubs/qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout)) associates
[`FermionicRegister`](../stubs/qiskit_fermions.circuit.FermionicRegister.html.md#qiskit_fermions.circuit.FermionicRegister) instances with
[`QuantumRegister`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) ones.

The way to configure this global setting, is by placing a [`F2QLayout`](../stubs/qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout) instance in the
`f2q_layout` field of the `property_set`. For more details
refer to [Layouting Passes](qiskit_fermions.transpiler.passes.html.md#qiskit-fermions-transpiler-passes-layout).

<a id="qiskit-fermions-transpiler-stage-synthesis"></a>

### Synthesis

At its core, the transpilation is handled by the [`F2QSynthesis`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis) transpiler pass. It is
conceptually similar to Qiskit’s [`HighLevelSynthesis`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.HighLevelSynthesis) pass, which
uses various plugins for transpiling high-level circuit instructions. For more details, refer to
the documentation of [`F2QSynthesis`](../stubs/qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis) directly.

How a given [`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) can be synthesized in terms of qubit-based operations will
depend on the particular gate type as well as the user-chosen fermion-to-qubit mapping. For more
details, refer to [Transpiler Pass Synthesis (F2Q)](qiskit_fermions.transpiler.passes.synthesis.html.md#qiskit-fermions-transpiler-passes-synthesis-plugins).

<a id="qiskit-fermions-transpiler-stage-qubit"></a>

### Qubit

At this point in the transpilation process, we have reached a
[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit) instance and can continue to use Qiskit’s transpilation
pipeline as one would usually.

#### HINT
Additional transpiler passes for optimizations on the qubit-level that take into account the
knowledge of a circuit originating from a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) may be added in the future!

<a id="qiskit-fermions-transpiler-stage-output"></a>

### Output

This stage implements effectively the reverse of [Input](#qiskit-fermions-transpiler-stage-input), by
calling the [`QuantumDAGToCircuit`](../stubs/qiskit_fermions.transpiler.QuantumDAGToCircuit.html.md#qiskit_fermions.transpiler.QuantumDAGToCircuit) transpiler pass.

## Pass Managers

All of the stages above are the default stages by how the [`presets`](qiskit_fermions.transpiler.presets.html.md#module-qiskit_fermions.transpiler.presets)
orchestrate the various `MultiStagePassManager`.

The individual stages can be either a single pass (when they change from one internal representation
(IR) to another, such as the [Input](#qiskit-fermions-transpiler-stage-input),
[Synthesis](#qiskit-fermions-transpiler-stage-synthesis), and
[Output](#qiskit-fermions-transpiler-stage-output) stages above) or a stage can be a
[`BasePassManager`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.BasePassManager) whose internal passes can be modified.

For the stages operating on fermionic circuits ([Optimization](#qiskit-fermions-transpiler-stage-optimization)
and [Layout](#qiskit-fermions-transpiler-stage-layout)), the type of passmanager to use is
[`FermionicPassManager`](../stubs/qiskit_fermions.transpiler.FermionicPassManager.html.md#qiskit_fermions.transpiler.FermionicPassManager).

### Conversion Passes

Some very basic conversion passes are provided directly by this module:

### Presets

For user convenience, the [`qiskit_fermions.transpiler.presets`](qiskit_fermions.transpiler.presets.html.md#module-qiskit_fermions.transpiler.presets) module provides a number of
functions for quickly building pre-defined transpilation pipelines.
