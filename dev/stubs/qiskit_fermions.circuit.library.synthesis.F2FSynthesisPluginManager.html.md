# F2FSynthesisPluginManager

### *class* F2FSynthesisPluginManager

A simple manager of the installed fermion-to-fermion synthesis plugins.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

All plugins that are registered under the `qiskit_fermions.circuit.synthesis`
[entry-point](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) will be
available to this class and get exposed by the [`plugins`](#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPluginManager.plugins) attribute. The name of an entry
point is the name of the gate class followed by the name of the method, separated by a dot, for
example `OrbitalRotation.Givens`. Below is the table from the `pyproject.toml` of this package
itself:

```toml
[project.entry-points."qiskit_fermions.circuit.synthesis"]
"Evolution.LieTrotter" = "qiskit_fermions.circuit.library.synthesis:FermionicLieTrotter"
"Evolution.SuzukiTrotter" = "qiskit_fermions.circuit.library.synthesis:FermionicSuzukiTrotter"
"Evolution.QDrift" = "qiskit_fermions.circuit.library.synthesis:FermionicQDrift"
"OrbitalRotation.Givens" = "qiskit_fermions.circuit.library.synthesis:FermionicGivensDecomposition"
"PrepareSlaterDeterminant.Givens" = "qiskit_fermions.circuit.library.synthesis:FermionicSlaterGivensDecomposition"

```

An overview of all builtin plugins is given [here](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#builtin-f2f-synthesis-plugins).

### Attributes

#### plugins *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]]*

A dictionary mapping gate names to their synthesis methods.

The keys of this dictionary are names of [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) classes. The values are the
list of installed [`F2FSynthesisPlugin`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin) methods that may be used to synthesize the
instances of that particular gate.

### Methods

#### method(op_name, method_name)

Returns the requested [`F2FSynthesisPlugin`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin) type.

* **Parameters:**
  * **op_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the gate whose plugin to return.
  * **method_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the synthesis plugin.
* **Returns:**
  The [`F2FSynthesisPlugin`](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin) class.
* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if no plugin is registered under that name.
* **Return type:**
  [type](https://docs.python.org/3/builtins/functions.html#type)[[*F2FSynthesisPlugin*](qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin.html.md#qiskit_fermions.circuit.library.synthesis.F2FSynthesisPlugin)]

#### method_names(op_name)

Returns the names of all installed plugin methods for a given gate.

* **Parameters:**
  **op_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the gate whose synthesis methods to return.
* **Returns:**
  The list of installed plugin method names.
* **Return type:**
  [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### op_names()

Returns the names of all gates for which any plugin is installed.

* **Return type:**
  [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]
