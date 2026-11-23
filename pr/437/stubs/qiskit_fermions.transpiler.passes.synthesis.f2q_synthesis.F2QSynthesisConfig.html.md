# F2QSynthesisConfig

### F2QSynthesisConfig

The dictionary type used to configure the [`F2QSynthesis.methods`](qiskit_fermions.transpiler.passes.F2QSynthesis.html.md#qiskit_fermions.transpiler.passes.F2QSynthesis.methods).

The keys of this dictionary must be names of [`FermionicInstruction`](qiskit_fermions.circuit.FermionicInstruction.html.md#qiskit_fermions.circuit.FermionicInstruction) circuit instructions.
The values can be one of four types:

1. `str`: the simplest scenario simply specifies the name of the plugin method to use for the
   corresponding key in this dictionary. The plugin is extracted from the
   [`F2QSynthesisPluginManager`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPluginManager) and its `__init__` method may **not** require any
   arguments.
2. `tuple[str, Sequence[Any]]`: the first `str` is the same as the above, and the second
   `Sequence[Any]` can be used to provide any positional arguments to the plugin’s `__init__`
   method.
3. `tuple[str, Sequence[Any], Mapping[str, Any]]`: the first two fields are the same as in 2. and
   the third `Mapping[str, Any]` can be used to provide any keyword arguments to the plugin’s
   `__init__` method.
4. an instance of an [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin), which is used as is. This is how a plugin that is
   not registered through an entry point enters the pass.

alias of [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`F2QSynthesisPlugin`](qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin.html.md#qiskit_fermions.transpiler.passes.synthesis.F2QSynthesisPlugin)]
