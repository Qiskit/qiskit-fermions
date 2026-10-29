# F2QSynthesis

### *class* F2QSynthesis(config=None)

A transpilation pass to map fermion-based circuit instructions to qubit-based ones.

Bases: [`GenericPass`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.GenericPass)[[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit), [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)]

This transpilation pass works similarly to Qiskit’s
[`HighLevelSynthesis`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.HighLevelSynthesis) pass; given an input
[`FermionicDAGCircuit`](qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit), it iterates the contained instructions and delegates the
translation to qubit-based instructions to matching [`methods`](#qiskit_fermions.transpiler.passes.F2QSynthesis.methods). The insertion of the
qubit-based circuit instructions into the output [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit) is also
left to the plugin method. This pass will merely have prepared the
[`QuantumRegister`](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#qiskit.circuit.QuantumRegister) according to the global transpilation
[`F2QLayout`](qiskit_fermions.transpiler.F2QLayout.html.md#qiskit_fermions.transpiler.F2QLayout) setting.

### Usage

There are two ways to configure the plugins used by this transpiler pass. The examples below
show the equivalent configuration to the [`generate_preset_jw_pass_manager()`](qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager.html.md#qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager).

1. Providing a `config` during initialization:
   ```pycon
   >>> from qiskit_fermions.mappers.library import jordan_wigner
   >>> from qiskit_fermions.transpiler import passes
   >>>
   >>> config = {
   ...     "Evolution": ("MapperFn", (jordan_wigner,)),
   ...     "FermionicMeasure": "TrivialOccupation",
   ...     "InitializeModes": "TrivialOccupation",
   ... }
   >>> synth = passes.F2QSynthesis(config)
   ```
2. Manually populating the [`methods`](#qiskit_fermions.transpiler.passes.F2QSynthesis.methods) attribute:
   ```pycon
   >>> from qiskit_fermions.mappers.library import jordan_wigner
   >>> from qiskit_fermions.transpiler import passes
   >>>
   >>> synth = passes.F2QSynthesis()
   >>> synth.methods["Evolution"] = passes.MapperFnEvolutionSynthesis(jordan_wigner)
   >>> synth.methods["FermionicMeasure"] = passes.TrivialOccupationFermionicMeasureSynthesis()
   >>> synth.methods["InitializeModes"] = passes.TrivialOccupationInitializeModesSynthesis()
   ```

   Through this manual approach it is also possible to inject custom
   [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) implementations without requiring them to be registered through
   an [entry-point](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).

Initializing this transpiler pass can be done with the arguments listed below.

* **Parameters:**
  **config** ([*F2QSynthesisConfig*](qiskit_fermions.transpiler.passes.synthesis.f2q_synthesis.F2QSynthesisConfig.html.md#qiskit_fermions.transpiler.passes.synthesis.f2q_synthesis.F2QSynthesisConfig) *|* *None*) – an optional dictionary to pre-populate the [`methods`](#qiskit_fermions.transpiler.passes.F2QSynthesis.methods) with plugins provided
  by the [`F2QSynthesisPluginManager`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager), or with plugin instances. See
  `F2QSynthesisConfig`.
* **Raises:**
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `config` names an instruction or a method for which no plugin is
        installed.
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `config` gives a value that is none of the forms of
        `F2QSynthesisConfig`.

### Attributes

#### methods *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [F2QSynthesisPlugin](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin)]*

A dictionary of fermion-to-qubit circuit instruction transpilation methods.

For this transpilation pass to have any effect, this dictionary must be populated with
instances of the [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) protocol. The keys of this dictionary
correspond to the `__name__` of a circuit instruction.

All available transpilation plugins are managed by the [`F2QSynthesisPluginManager`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager).

#### NOTE
This dictionary is **empty** by default! You can pre-populate it with plugins provided by
the [`F2QSynthesisPluginManager`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager) by providing a `config` argument during
initialization of this transpiler pass instance.

### Methods

#### run(dag)

Runs this transpilation pass.

* **Parameters:**
  **dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the input circuit with fermion-based instructions. Only
  [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) with [`FermionicInstruction`](qiskit_fermions.circuit.FermionicInstruction.html.md#qiskit_fermions.circuit.FermionicInstruction) instances
  as their [`op`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode#op) are supported (that is every
  [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) as well as [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure)), plus
  `Barrier`, which is mapped onto every qubit of each
  register it touched.
* **Returns:**
  The output circuit with qubit-based instructions.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – when a [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) is encountered whose
        [`op`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode#op) is neither of type
        [`FermionicInstruction`](qiskit_fermions.circuit.FermionicInstruction.html.md#qiskit_fermions.circuit.FermionicInstruction) nor a `Barrier`.
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – when a [`FermionicInstruction`](qiskit_fermions.circuit.FermionicInstruction.html.md#qiskit_fermions.circuit.FermionicInstruction) type is encountered for which no
        translation plugin is present in [`methods`](#qiskit_fermions.transpiler.passes.F2QSynthesis.methods).
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
