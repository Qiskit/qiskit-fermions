# Selector

### *class* Selector(\*args, \*\*kwargs)

Chooses one of the variants, or returns `None` to decline the gate.

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A selector is called as `selector(variants, problem, context)` once per gate, with every
[`Variant`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant) the candidates built, in tie-break order (the order of the plugin’s
candidates, then each candidate’s own order), the gate’s [`Problem`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem) and its
[`Context`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context). It returns one of the variants, or `None` to decline the gate. Any callable
with this signature is a selector, a plain function included. [`MinCost`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost) is the built-in
one.
