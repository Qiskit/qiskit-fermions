# FermionicDAGCircuit

### FermionicDAGCircuit *: [TypeAlias](https://docs.python.org/3/library/typing.html#typing.TypeAlias)* *= <class 'qiskit._accelerate.circuit.DAGCircuit'>*

A type alias of [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit).

Although this does not really give us any functional guarantees, it serves to better document the
API contract that instances of this type only contain circuit instructions of type
[`FermionicInstruction`](qiskit_fermions.circuit.FermionicInstruction.html.md#qiskit_fermions.circuit.FermionicInstruction) (i.e. any [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) or a [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure)) or
`Barrier`.
