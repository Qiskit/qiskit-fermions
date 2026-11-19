# FermionicLieTrotter

### *class* FermionicLieTrotter(reps=1)

The first-order Lie-Trotter product formula, applied in fermionic space.

Bases: [`FermionicSuzukiTrotter`](qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter)

Given an [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate with the evolution time $t$, this synthesizes the
approximation

$$
e^{-i t \sum_j H_j} \approx \prod_j e^{-i t H_j},
$$

emitting one [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate per factor $H_j$, each evolving for the whole time
$t$ – or, with [`reps`](#qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter.reps) repetitions, for $t/\texttt{reps}$
apiece across that many sweeps.

The operator is split into the $H_j$, and each factor narrowed to the modes it actually
touches, exactly as described for [`FermionicSuzukiTrotter`](qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter).

First order is the degenerate case of the Suzuki recursion, so this is implemented as a
[`FermionicSuzukiTrotter`](qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter) with the order fixed to `1`; the two are interchangeable at
equal [`reps`](#qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter.reps). It keeps a name of its own because it is the default
synthesis of an [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate, and because a plain first-order product formula is more
recognizable under that name than as an argument to a higher-order one.

#### NOTE
This product formula is exact only when the factors mutually commute. Grouping an operator so
that each group collects mutually commuting terms therefore both shortens the circuit and
reduces the Trotter error. Where the factors do not commute, an even
[`order`](qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.order) reduces the error further at the same grouping.

Note also that applying it at all is optional: leaving the gate undecomposed and letting the
fermion-to-qubit stage handle the whole operator incurs no Trotter error at this level. See
[`synthesis`](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis).

The Hermiticity requirement on the individual factors, described in the
[`FermionicSuzukiTrotter`](qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicSuzukiTrotter) documentation, applies here too.

Initializing an instance of this synthesis method can be done with the argument below.

* **Parameters:**
  **reps** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of times to repeat the formula, each repetition evolving for
  `time / reps`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `reps` is not positive.

### Attributes

#### order

The order of the product formula.

#### reps

The number of times the product formula is repeated.

#### merge_adjacent

Whether neighboring gates of the same factor are fused into one.

The palindromic structure makes the last factor of one repetition (and of each sub-sweep of
a higher order) the first of the next. Their exponentials combine exactly, as
$e^{-i a H} e^{-i b H} = e^{-i (a + b) H}$, so fusing them shortens the circuit without
changing the product formula. Disable this to obtain the textbook gate sequence.

This only sees the seams created by a single synthesis. Merging across gate boundaries is
the job of [`MergeEvolutions`](qiskit_fermions.transpiler.passes.MergeEvolutions.html.md#qiskit_fermions.transpiler.passes.MergeEvolutions).

### Inherited Methods

#### synthesize(gate)

Synthesizes the gate into the ordered factors of the product formula.

See the class documentation for the formula this implements.

* **Parameters:**
  **gate** ([*Evolution*](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)) – the gate to synthesize.
* **Returns:**
  A [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) holding one narrowed [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate per factor of
  the formula. A factor may appear more than once, at different evolution times.
* **Return type:**
  [FermionicCircuit](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)
