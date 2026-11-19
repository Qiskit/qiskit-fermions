# GivensDecompositionSlaterDeterminantSynthesis

### *class* GivensDecompositionSlaterDeterminantSynthesis(, minimize_2q_gate_count=False)

**DEPRECATED** The former plugin for transpiling a [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) into Givens rotations.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This is the [`FermionicSlaterGivensDecomposition`](qiskit_fermions.circuit.library.synthesis.FermionicSlaterGivensDecomposition.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSlaterGivensDecomposition) followed by the Jordan-Wigner mapping
of its gates.

Initializing this plugin can be done with the argument below.

#### Deprecated
Deprecated since version 0.4.0: The class `qiskit_fermions.transpiler.passes.synthesis._deprecated_givens.GivensDecompositionSlaterDeterminantSynthesis` is deprecated as of qiskit-fermions 0.4.0. It will be removed no earlier than 3 months after the release date. Break the gate down with F2FSynthesis({‘PrepareSlaterDeterminant’: ‘Givens’}) in the optimization stage instead, which works with any encoding.

* **Parameters:**
  **minimize_2q_gate_count** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – no longer supported. The peeling of unit-orbitals that this
  enabled was removed, so `True` has no effect and warns.

### Attributes

#### minimize_2q_gate_count

Always `False`, see `__init__`.

### Methods

#### run(in_node, out_dag, , f2q_layout)

Runs this transpilation plugin.

* **Parameters:**
  * **in_node** ([*DAGOpNode*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode)) – the input fermion-based circuit instruction, a
    [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant).
  * **out_dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the output qubit-based circuit.
  * **f2q_layout** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *,* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *]*) – the global transpilation [`F2QLayout`](qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout)
    setting.
* **Return type:**
  None
