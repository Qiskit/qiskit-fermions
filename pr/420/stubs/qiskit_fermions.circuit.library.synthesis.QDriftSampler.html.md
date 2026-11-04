# QDriftSampler

### *class* QDriftSampler(gate, num_terms, weights, rng)

The draws of the qDRIFT protocol for one [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This is what [`FermionicQDrift.sampler()`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift.sampler) returns. It holds the distribution derived from
the gate, and exists so that a caller that needs to look at each draw (the
[`QDriftTrotterization`](qiskit_fermions.transpiler.passes.QDriftTrotterization.html.md#qiskit_fermions.transpiler.passes.QDriftTrotterization) pass, when it rejects draws) can take them one at a time, whereas
[`FermionicQDrift.synthesize()`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift.synthesize) takes them all at once.

Initializing a sampler can be done with the arguments below.

* **Parameters:**
  * **gate** ([*Evolution*](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)) – the gate to sample.
  * **num_terms** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of gates the whole sample holds.
  * **weights** (*np.ndarray* *|* *None*) – the sampling weights, or `None` to derive them from the operator of `gate`.
  * **rng** (*np.random.Generator*) – the random number generator to draw from.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `weights` does not hold one entry per group (or term) of the operator.

### Attributes

#### time

The evolution time $\delta$ of every sampled gate.

### Methods

#### draw()

Draws a single gate.

* **Returns:**
  The drawn atomic [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate.
* **Return type:**
  [Evolution](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)

#### sample(size=None)

Draws `size` gates in a single batched call.

* **Parameters:**
  **size** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* *None*) – the number of gates to draw. Defaults to `num_terms`.
* **Returns:**
  The drawn atomic [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates, in the order of the draws.
* **Return type:**
  [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[Evolution](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)]
