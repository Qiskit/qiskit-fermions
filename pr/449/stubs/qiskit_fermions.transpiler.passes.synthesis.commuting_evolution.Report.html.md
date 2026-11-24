# Report

### *class* Report(problem, context, variants, chosen, declined)

What the plugin did with one gate, handed to its `callback`.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

### Attributes

* **Parameters:**
  * **problem** ([*Problem*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem) *|* *None*)
  * **context** ([*Context*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context) *|* *None*)
  * **variants** ([*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) *[*[*Variant*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant) *]*)
  * **chosen** ([*Variant*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant) *|* *None*)
  * **declined** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *|* *None*)

#### problem *: [Problem](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The gate’s problem, or `None` if the gate was declined before one was built.

#### context *: [Context](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The gate’s context, or `None` likewise.

#### variants *: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[Variant](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant)]*

Every circuit the candidates built, in tie-break order.

#### chosen *: [Variant](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The selector’s choice, or `None` if the gate was declined.

#### declined *: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Why the gate was declined, or `None` if it was synthesized.
