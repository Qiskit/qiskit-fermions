# MergeOrbitalRotations

### *class* MergeOrbitalRotations

A transpilation pass merging runs of consecutive orbital rotations into a single one.

Bases: [`GenericPass`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.GenericPass)[[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit), [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)]

Two [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) gates applied back-to-back on the same modes compose into a single
orbital rotation whose `rotation_unitary` is the matrix product of the two. This pass detects
maximal runs of such consecutive rotations in a [`FermionicDAGCircuit`](qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit) and rewrites each
into one [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation), so the synthesis stage lowers a single (still square)
decomposition instead of one per gate in the run.

A “run” is a maximal chain of [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) nodes acting on the *same* mode set with
nothing else intervening on those wires. Following the [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) convention
$a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j$, applying a rotation $U_1$ and then a
rotation $U_2$ maps $a^\dagger_i \mapsto \sum_j (U_2 U_1)_{ji} a^\dagger_j$, so the
run’s rotations are multiplied in circuit order (later rotation on the left) to form the merged
`rotation_unitary`. Because the merged gate implements exactly that composed basis change, the
rewrite leaves the simulated state vector unchanged.

Only rotations on the *identical* mode set are merged: two rotations on different (even
overlapping) mode sets, or rotations separated by any other operation on a shared wire, break the
run and are left untouched. This is the shape produced, e.g., by two adjacent per-spin rotations
on the same spin half, or by consecutive [`UCJ`](qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ) layers whose trailing and leading orbital
rotations meet on the same modes.

#### CAUTION
This is an early development prototype. Beware of changes to its interface without warning
during the pre-release development of this package.

#### SEE ALSO
[`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation).

### Methods

#### run(dag)

Runs this transpilation pass.

Collects every maximal run of consecutive [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) gates on the same modes
and replaces each run of length two or more with a single [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) whose
`rotation_unitary` is the run’s rotations multiplied in circuit order. Runs of a single
rotation (and all other nodes) are left untouched. The input DAG is modified in place.

* **Parameters:**
  **dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the input circuit with fermion-based instructions. Only
  [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) with [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) instances as their
  [`op`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode#op) are supported.
* **Returns:**
  The output circuit which is still acting on a fermionic register.
* **Return type:**
  [*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)

### Inherited Methods

#### execute(passmanager_ir, state, callback=None)

Execute optimization task for input Qiskit IR.

* **Parameters:**
  * **passmanager_ir** (*IR*) – Qiskit IR to optimize.
  * **state** ([*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)) – State associated with workflow execution by the pass manager itself.
  * **callback** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[*[*Task*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.Task) *,* *IR_OUT* *,* [*PropertySet*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PropertySet) *,* [*float*](https://docs.python.org/3/builtins/functions.html#float) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *,* *None* *]*  *|* *None*) – A callback function which is called per execution of optimization task.
* **Returns:**
  Optimized Qiskit IR and state of the workflow.
* **Return type:**
  [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[*IR_OUT*, [*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)]

#### name()

Name of the pass.

* **Return type:**
  [str](https://docs.python.org/3/builtins/stdtypes.html#str)

#### update_status(state, run_state)

Update workflow status.

* **Parameters:**
  * **state** ([*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)) – Pass manager state to update.
  * **run_state** (*RunState*) – Completion status of current task.
* **Returns:**
  Updated pass manager state.
* **Return type:**
  [*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)
