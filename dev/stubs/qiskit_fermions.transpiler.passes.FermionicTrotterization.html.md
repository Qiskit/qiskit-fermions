# FermionicTrotterization

### *class* FermionicTrotterization(synthesis, , apply=True, filter=None)

**DEPRECATED** The former pass to Trotterize [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates.

Bases: [`F2FSynthesis`](qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis)

This is [`F2FSynthesis`](qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis) restricted to the [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate: the `synthesis`
argument is the method for that gate.

Initializing this transpiler pass can be done with the arguments listed below.

#### Deprecated
Deprecated since version 0.4.0: The class `qiskit_fermions.transpiler.passes.optimization._deprecated_trotterization.FermionicTrotterization` is deprecated as of qiskit-fermions 0.4.0. It will be removed no earlier than 3 months after the release date. Use F2FSynthesis({“Evolution”: synthesis}) instead.

* **Parameters:**
  * **synthesis** ([*F2FSynthesisPlugin*](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin)) – the synthesis method for [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates.
  * **apply** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – see [`F2FSynthesis`](qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis).
  * **filter** (*Callable* *[* *[**DAGOpNode* *]* *,* [*bool*](https://docs.python.org/3/builtins/functions.html#bool) *]*  *|* *None*) – see [`F2FSynthesis`](qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis).

### Attributes

#### methods *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [F2FSynthesisPlugin](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin)]*

A dictionary of the fermion-to-fermion synthesis methods, by gate.

The keys of this dictionary correspond to the `__name__` of a [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate), and
the values are instances of the [`F2FSynthesisPlugin`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin) protocol. A gate that is not a
key is not touched by this pass, so the dictionary is **empty** by default, which makes the
pass do nothing. Pre-populate it with plugins provided by the [`F2FSynthesisPluginManager`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPluginManager.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPluginManager)
through the `config` argument.

#### apply

Whether to expand each selected gate into the factors its plugin produces.

When this is `False`, the pass only *selects* the method: it tags each selected gate with
its plugin (see [`FermionicGate.with_synthesis()`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.with_synthesis)), except an [`atomic`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.atomic)
gate, which is left as it is, and leaves the expansion to whatever
builds the gate’s definition later, such as a [`decompose()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.decompose) call. Note
that a gate which is never expanded reaches the fermion-to-qubit stage whole, where it is
mapped without its [`synthesis`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.synthesis) ever being read.

#### filter

The predicate selecting which nodes to break down.

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

#### run(dag)

Runs this transpilation pass.

Every node whose gate type is in [`methods`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.methods) and that is accepted by [`filter`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.filter) is
replaced by the gates that its plugin produces, or (when [`apply`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.apply) is `False`) by an
equivalent gate merely carrying the plugin. All other nodes are left untouched. The input DAG
is modified in place.

* **Parameters:**
  **dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the input circuit with fermion-based instructions. Only
  [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) with [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) instances as their
  [`op`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode#op) are supported, plus
  `Barrier` and [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure), which are
  carried through untouched (a measurement keeps the classical bit it writes into).
* **Returns:**
  The output circuit which is still acting on a fermionic register.
* **Return type:**
  [*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)

#### update_status(state, run_state)

Update workflow status.

* **Parameters:**
  * **state** ([*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)) – Pass manager state to update.
  * **run_state** (*RunState*) – Completion status of current task.
* **Returns:**
  Updated pass manager state.
* **Return type:**
  [*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)
