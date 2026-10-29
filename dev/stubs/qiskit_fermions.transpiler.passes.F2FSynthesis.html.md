# F2FSynthesis

### *class* F2FSynthesis(config=None, , apply=True, filter=None)

A transpilation pass breaking gates down into smaller ones in fermionic space.

Bases: [`GenericPass`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.GenericPass)[[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit), [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)]

Some gates can be rewritten into a circuit of smaller fermionic gates before they reach the
fermion-to-qubit stage, which then maps and synthesizes those. This pass does that for the gates
named in [`methods`](#qiskit_fermions.transpiler.passes.F2FSynthesis.methods), with the plugin that [`methods`](#qiskit_fermions.transpiler.passes.F2FSynthesis.methods) names for each, so the choice is
made once for a whole transpilation pipeline instead of gate by gate. It is the fermionic
counterpart of [`F2QSynthesis`](qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis), one stage earlier: where that pass maps every instruction
to qubits, this one rewrites some gates into others and leaves the rest as they are.

```python
pm.optimization = FermionicPassManager(
    [
        MergeOrbitalRotations(),
        MergeSlaterDeterminantPreparation(),
        F2FSynthesis({"OrbitalRotation": "Givens", "PrepareSlaterDeterminant": "Givens"}),
    ]
)
```

### Usage

There are two ways to configure the plugins used by this transpiler pass:

1. Providing a `config` during initialization:
   ```pycon
   >>> from qiskit_fermions.transpiler import passes
   >>>
   >>> config = {"Evolution": ("SuzukiTrotter", (), {"order": 2})}
   >>> synth = passes.F2FSynthesis(config)
   ```
2. Manually populating the [`methods`](#qiskit_fermions.transpiler.passes.F2FSynthesis.methods) attribute:
   ```pycon
   >>> from qiskit_fermions.circuit.library import synthesis
   >>> from qiskit_fermions.transpiler import passes
   >>>
   >>> synth = passes.F2FSynthesis()
   >>> synth.methods["Evolution"] = synthesis.FermionicSuzukiTrotter(order=2)
   ```

   Through this manual approach it is also possible to inject custom
   [`F2FSynthesisPlugin`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin) implementations without requiring them to be registered through
   an [entry-point](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).

It belongs in the `optimization` stage, ahead of the layout and the synthesis, and after the
passes that consume these gates as such. Placing it earlier would break the gates up before those
passes can fuse them, which silently gives up the cheaper Slater-determinant decomposition (see
[`MergeSlaterDeterminantPreparation`](qiskit_fermions.transpiler.passes.MergeSlaterDeterminantPreparation.html.md#qiskit_fermions.transpiler.passes.MergeSlaterDeterminantPreparation)).

Each selected gate is replaced by the circuit that its plugin produces. A gate whose type is not a
key of [`methods`](#qiskit_fermions.transpiler.passes.F2FSynthesis.methods) is left untouched, as is any gate that is [`atomic`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.atomic)
(also with `apply=False`: it is not even tagged with the plugin)
and any node rejected by an optional [`filter`](#qiskit_fermions.transpiler.passes.F2FSynthesis.filter). A gate that has no synthesis at all, such as
an [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes), passes through no matter what.

#### IMPORTANT
What the replacement is equivalent to is specified by the gate and its plugin, not by this
pass: usually it implements the same unitary (or approximates it), but a gate may specify
less, and then only a circuit that meets that weaker specification may use this pass. Consult
the documentation of the gates you configure, and see [`F2FSynthesisPlugin`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin).

#### NOTE
Running the pass again is harmless: the factors a plugin emits are
[`atomic`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.atomic), and an atomic gate is a terminal factor which this pass leaves
in place rather than splitting further. The same exemption makes the pass a fixed point over
the output of [`QDriftTrotterization`](qiskit_fermions.transpiler.passes.QDriftTrotterization.html.md#qiskit_fermions.transpiler.passes.QDriftTrotterization), whose sampled gates are atomic because the random
draw *is* the Trotterization it performs.

#### NOTE
Set [`apply`](#qiskit_fermions.transpiler.passes.F2FSynthesis.apply) to `False` to only *select* the method and leave the expansion to
something else, such as a [`decompose()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.decompose) call. Be aware that a gate which
is never expanded reaches the fermion-to-qubit stage whole, where it is mapped without its
[`synthesis`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.synthesis) ever being read, and every method then produces identical
output.

#### CAUTION
Not every method suits every operator. An [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) whose operator groups all
mutually commute (the diagonal-Coulomb operators of a [`UCJ`](qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ), for example) is
synthesized exactly at any order, so a higher order only adds depth. Use [`filter`](#qiskit_fermions.transpiler.passes.F2FSynthesis.filter) to
restrict the pass to the gates that benefit.

Expanding an operator that carries no [`groups`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups)
splits it term by term, and a lone $a^\dagger_i a_j$ is not Hermitian, so its
exponential is not unitary. Nothing complains until the mapped operator reaches
[`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate), which raises `ValueError: Operator
contains complex coefficients, which are not supported`. Assigning conjugate-paired groups
is what makes each factor Hermitian.

Initializing this transpiler pass can be done with the arguments listed below.

* **Parameters:**
  * **config** ([*F2FSynthesisConfig*](qiskit_fermions.transpiler.passes.optimization.f2f_synthesis.F2FSynthesisConfig.html.md#qiskit_fermions.transpiler.passes.optimization.f2f_synthesis.F2FSynthesisConfig) *|* *None*) – an optional dictionary to pre-populate the [`methods`](#qiskit_fermions.transpiler.passes.F2FSynthesis.methods) with plugins provided
    by the [`F2FSynthesisPluginManager`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPluginManager.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPluginManager). See `F2FSynthesisConfig`.
  * **apply** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to expand each selected gate into the factors its plugin produces. When
    `False`, the gate is only tagged with the plugin (unless it is atomic, which is
    never touched) and something else has to expand it later. See [`apply`](#qiskit_fermions.transpiler.passes.F2FSynthesis.apply).
  * **filter** (*Callable* *[* *[**DAGOpNode* *]* *,* [*bool*](https://docs.python.org/3/builtins/functions.html#bool) *]*  *|* *None*) – an optional predicate deciding which nodes to break down. It is called with the
    [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) of a gate whose type is in [`methods`](#qiskit_fermions.transpiler.passes.F2FSynthesis.methods),
    and the node is left untouched unless it returns `True`. If `None` (the default),
    every such node is selected.
* **Raises:**
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `config` names a gate or a method for which no plugin is installed.
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `config` gives a value that is none of the forms of
        `F2FSynthesisConfig`.

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

### Methods

#### run(dag)

Runs this transpilation pass.

Every node whose gate type is in [`methods`](#qiskit_fermions.transpiler.passes.F2FSynthesis.methods) and that is accepted by [`filter`](#qiskit_fermions.transpiler.passes.F2FSynthesis.filter) is
replaced by the gates that its plugin produces, or (when [`apply`](#qiskit_fermions.transpiler.passes.F2FSynthesis.apply) is `False`) by an
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
