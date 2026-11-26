# GivensDecompositionOrbitalRotationSynthesis

### *class* GivensDecompositionOrbitalRotationSynthesis

**DEPRECATED** The former plugin for transpiling an [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) into Givens rotations.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This is the [`FermionicGivensDecomposition`](qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition.html.md#qiskit_fermions.circuit.library.synthesis.FermionicGivensDecomposition) followed by the Jordan-Wigner mapping of its
factors.

Initializing this plugin takes no arguments.

#### Deprecated
Deprecated since version 0.4.0: The class `qiskit_fermions.transpiler.passes.synthesis._deprecated_givens.GivensDecompositionOrbitalRotationSynthesis` is deprecated as of qiskit-fermions 0.4.0. It will be removed no earlier than 3 months after the release date. Break the gate down with F2FSynthesis({‘OrbitalRotation’: ‘Givens’}) in the optimization stage instead, which works with any encoding.

### Methods

#### run(in_node, out_dag, , f2q_layout)

Runs this transpilation plugin.

* **Parameters:**
  * **in_node** ([*DAGOpNode*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode)) – the input fermion-based circuit instruction, an [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation).
  * **out_dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the output qubit-based circuit.
  * **f2q_layout** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *,* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *]*) – the global transpilation [`F2QLayout`](qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout)
    setting.
* **Return type:**
  None
