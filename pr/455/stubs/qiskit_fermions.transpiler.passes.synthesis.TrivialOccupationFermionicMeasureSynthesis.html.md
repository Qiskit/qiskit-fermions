# TrivialOccupationFermionicMeasureSynthesis

### *class* TrivialOccupationFermionicMeasureSynthesis

A [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) for transpiling [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure) with trivial occupation.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Under an occupation-basis encoding, the occupation of fermionic mode $i$ is the
computational-basis value of the single qubit that mode is mapped onto. A mode measurement
therefore lowers to exactly one qubit [`Measure`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.Measure), writing into the
very classical bit the [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure) was given.

#### WARNING
This transpilation pass plugin makes the following assumptions:

- an occupation-basis encoding (like Jordan-Wigner)
- a trivial fermion-to-qubit layout (i.e. no change in their register lengths)
- a 1-to-1 mapping of fermionic mode indices to qubit indices

### Methods

#### run(in_node, out_dag, , f2q_layout)

Runs this transpilation plugin.

* **Parameters:**
  * **in_node** ([*DAGOpNode*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode)) – the input fermion-based circuit instruction. When this plugin gets called, the
    `in_node.op` attribute must be of type [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure).
  * **out_dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the output qubit-based circuit.
  * **f2q_layout** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *,* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *]*) – the global transpilation [`F2QLayout`](qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout)
    setting.
* **Return type:**
  None

#### SEE ALSO
The documentation of [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) for more detailed explanations of the
arguments.

* **Raises:**
  * [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) – when `in_node` acts on fermionic modes that are spread across
        multiple [`FermionicRegister`](qiskit_fermions.circuit.FermionicRegister.html.md#qiskit_fermions.circuit.FermionicRegister) instances.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – when the mapped qubit register does not have the same length as the fermionic
        mode register, in which case a mode has no single qubit whose measurement would yield
        its occupation.
* **Parameters:**
  * **in_node** ([*DAGOpNode*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode))
  * **out_dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit))
  * **f2q_layout** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *,* [*QuantumRegister*](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) *]*)
* **Return type:**
  None
