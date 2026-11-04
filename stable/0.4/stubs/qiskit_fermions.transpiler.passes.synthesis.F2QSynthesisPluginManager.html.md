# F2QSynthesisPluginManager

### *class* F2QSynthesisPluginManager

A simple manager of the installed fermion-to-qubit synthesis plugins.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

All plugins that are registered under the `qiskit_fermions.transpiler.synthesis`
[entry-point](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) will be
available to this class and get exposed by the [`plugins`](#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager.plugins) attribute. Below is an example
from the `pyproject.toml` of this package itself:

```toml
[project.entry-points."qiskit_fermions.transpiler.synthesis"]
"Evolution.MapperFn" = "qiskit_fermions.transpiler.passes.synthesis:MapperFnEvolutionSynthesis"
"InitializeModes.TrivialOccupation" = "qiskit_fermions.transpiler.passes.synthesis:TrivialOccupationInitializeModesSynthesis"
"FermionicMeasure.TrivialOccupation" = "qiskit_fermions.transpiler.passes.synthesis:TrivialOccupationFermionicMeasureSynthesis"
# DEPRECATED all of the following entry-points are deprecated and subject to removal in a future release
"OrbitalRotation.GivensDecomposition" = "qiskit_fermions.transpiler.passes.synthesis:GivensDecompositionOrbitalRotationSynthesis"
"PrepareSlaterDeterminant.GivensDecompositionSlater" = "qiskit_fermions.transpiler.passes.synthesis:GivensDecompositionSlaterDeterminantSynthesis"

```

An overview of all builtin plugins is given [here](../pydoc/qiskit_fermions.transpiler.passes.synthesis.html.md#builtin-f2q-synthesis-plugins).

### Attributes

#### plugins *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]]*

A dictionary mapping circuit instruction names to their synthesis methods.

The keys of this dictionary are names of [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) classes. The values are the
list of installed [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) methods that may be used to synthesize the
instances of that particular gate to its qubit form.

### Methods

#### method(op_name, method_name)

Returns the requested [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) type.

* **Parameters:**
  * **op_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the operation whose plugin to return.
  * **method_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the synthesis plugin.
* **Returns:**
  The [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin) class.
* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if no plugin is registered under that name.
* **Return type:**
  [type](https://docs.python.org/3/builtins/functions.html#type)[[*F2QSynthesisPlugin*](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin)]

#### method_names(op_name)

Returns the names of all installed plugin methods for a given circuit instruction.

* **Parameters:**
  **op_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the operation whose synthesis methods to return.
* **Returns:**
  The list of installed plugin method names.
* **Return type:**
  [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### op_names()

Returns the names of all circuit instructions for which any plugin is installed.

* **Return type:**
  [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]
