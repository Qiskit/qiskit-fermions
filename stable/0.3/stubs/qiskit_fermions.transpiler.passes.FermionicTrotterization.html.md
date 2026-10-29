# FermionicTrotterization

### *class* FermionicTrotterization(synthesis, , apply=True, filter=None)

A transpilation pass Trotterizing [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates in fermionic space.

Bases: [`GenericPass`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.GenericPass)[[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit), [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)]

An [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate carries the synthesis method with which it gets decomposed in fermionic
space (see [`Evolution.synthesis`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.synthesis)). Setting it per gate means threading the choice through
everything that constructs one – including [`UCC`](qiskit_fermions.circuit.library.UCC.html.md#qiskit_fermions.circuit.library.UCC) and [`UCJ`](qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ), which build their own
internally. This pass applies one method to every [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) in a circuit instead, so the
choice can be made once for a whole transpilation pipeline:

```python
pm.optimization = FermionicPassManager(
    [FermionicTrotterization(FermionicSuzukiTrotter(order=2, reps=4))]
)
```

Each selected gate is replaced by the factors its method emits, so the pass both chooses the
formula and carries it out. Nodes that are not [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates are left untouched, as
are those rejected by an optional [`filter`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.filter).

#### NOTE
Running the pass again is harmless: the factors a synthesis method emits are
[`Evolution.atomic`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.atomic), and an atomic gate is a terminal factor which this pass leaves in
place rather than splitting further. The same exemption makes the pass a fixed point over the
output of [`QDriftTrotterization`](qiskit_fermions.transpiler.passes.QDriftTrotterization.html.md#qiskit_fermions.transpiler.passes.QDriftTrotterization), whose sampled gates are atomic because the random
draw *is* the Trotterization it performs.

#### NOTE
Set [`apply`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.apply) to `False` to only *select* the method and leave the expansion to
something else, such as Qiskit’s [`Decompose`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.Decompose) or a
[`decompose()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.decompose) call. Be aware that a gate which is never expanded
reaches the fermion-to-qubit stage whole, where it is mapped without
[`Evolution.synthesis`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.synthesis) ever being read – and every synthesis method then produces
identical output.

#### CAUTION
Not every synthesis method suits every operator. An [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) whose operator groups
all mutually commute (the diagonal-Coulomb operators of a [`UCJ`](qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ), for example) is
synthesized exactly at any order, so a higher order only adds depth. Use [`filter`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.filter) to
restrict the pass to the gates that benefit.

Expanding an operator that carries no [`groups`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups)
splits it term by term, and a lone $a^\dagger_i a_j$ is not Hermitian, so its
exponential is not unitary. Nothing complains until the mapped operator reaches
[`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate), which raises `ValueError: Operator
contains complex coefficients, which are not supported`. Assigning conjugate-paired groups
is what makes each factor Hermitian.

Initializing this transpiler pass can be done with the arguments listed below.

* **Parameters:**
  * **synthesis** ([*FermionicEvolutionSynthesis*](qiskit_fermions.circuit.library.synthesis.FermionicEvolutionSynthesis.html.md#qiskit_fermions.circuit.library.synthesis.FermionicEvolutionSynthesis)) – the fermion-to-fermion synthesis method to apply to the [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)
    gates of the circuit.
  * **apply** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to expand each selected gate into the factors `synthesis` produces.
    When `False`, the gate is only tagged with `synthesis` and something else has to
    expand it later. See [`apply`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.apply).
  * **filter** (*Callable* *[* *[**DAGOpNode* *]* *,* [*bool*](https://docs.python.org/3/builtins/functions.html#bool) *]*  *|* *None*) – an optional predicate deciding which [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) nodes to apply
    `synthesis` to. It is called with the [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) and the
    node is left untouched unless it returns `True`. If `None` (the default), every
    [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) node is selected.

### Attributes

#### synthesis

The fermion-to-fermion synthesis method applied to the selected gates.

#### apply

Whether to expand each selected gate into the factors [`synthesis`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.synthesis) produces.

When this is `False`, the pass only *selects* the method: it tags each selected gate with
[`synthesis`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.synthesis) and leaves the expansion to whatever builds the gate’s definition later,
such as Qiskit’s [`Decompose`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.Decompose) or a
[`decompose()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.decompose) call. Note that a gate which is never expanded reaches
the fermion-to-qubit stage whole, where it is mapped without [`Evolution.synthesis`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.synthesis)
ever being read, and every synthesis method then produces identical output.

#### filter

The predicate selecting which [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) nodes to apply [`synthesis`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.synthesis) to.

### Methods

#### run(dag)

Runs this transpilation pass.

Every [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) node accepted by [`filter`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.filter) is replaced by the factors that
[`synthesis`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.synthesis) produces, or (when [`apply`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.apply) is `False`) by an equivalent gate
merely carrying [`synthesis`](#qiskit_fermions.transpiler.passes.FermionicTrotterization.synthesis). All other nodes are left untouched. The input DAG is
modified in place.

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
