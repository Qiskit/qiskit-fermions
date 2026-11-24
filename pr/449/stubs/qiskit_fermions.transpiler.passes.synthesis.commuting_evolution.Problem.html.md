# Problem

### *class* Problem(x, z, coeffs, time, num_qubits)

One group of mutually commuting Pauli terms to be exponentiated: *what* is synthesized.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

It holds the terms as a matrix of X and Z bits with one row per term, and their coefficients.
The candidates work on these arrays: conjugation by a Clifford is arithmetic over GF(2) on the
columns of the bit matrix. Plain arrays also let a candidate be implemented in another language
against the same data. [`operator`](#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.operator) gives the terms as a
[`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable).

#### SEE ALSO
[`Context`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context), which holds *where* the circuit will run: what is known about the device,
and the pairs of qubits the user allows. A candidate receives both.

### Attributes

* **Parameters:**
  * **x** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray))
  * **z** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray))
  * **coeffs** ([*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray))
  * **time** ([*float*](https://docs.python.org/3/builtins/functions.html#float))
  * **num_qubits** ([*int*](https://docs.python.org/3/builtins/functions.html#int))

#### operator

The terms as a [`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable).

It is built from the arrays when first asked for.

#### paulis

The terms as a [`PauliList`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.PauliList) without phases.

It serves to conjugate the terms by Cliffords.

#### x *: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)*

The X bits of the terms (0/1, terms by qubits).

#### z *: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)*

The Z bits of the terms. A qubit with both bits set carries a Y.

#### coeffs *: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)*

The real coefficients of the terms.

#### time *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The evolution time `t` of `exp(-i t H)`.

#### num_qubits *: [int](https://docs.python.org/3/builtins/functions.html#int)*

The number of qubits.

### Methods

#### *classmethod* from_gate(gate)

The problem of a gate, or `None` if the gate is not one this plugin handles.

* **Parameters:**
  **gate** ([*Operation*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Operation)) – the operation the high-level synthesis hands to a plugin.
* **Returns:**
  The problem of a [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate) over a single
  operator with a real time, whose coefficients are real and whose terms mutually commute.
  `None` if the operation is not such a gate (a different operation, a list of
  operators, or a time that is not a plain real number).
* **Raises:**
  **Declined** – if the gate is a Pauli evolution of the right form, but its coefficients are
      complex or its terms do not commute.
* **Return type:**
  [*Problem*](#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem) | None

#### *classmethod* from_operator(operator, time)

The problem of `exp(-i t H)` for an operator whose terms commute and are real.

* **Parameters:**
  * **operator** ([*SparseObservable*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) *|* [*SparsePauliOp*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparsePauliOp)) – the terms. Projectors of a [`SparseObservable`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparseObservable) are
    expanded into Paulis, the terms are simplified, and the imaginary parts of the
    coefficients are dropped.
  * **time** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – the evolution time.
* **Returns:**
  The problem.
* **Return type:**
  [*Problem*](#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem)
