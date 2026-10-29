# Evolution

### *class* Evolution(num_modes, operator, time=1.0, , synthesis=None, atomic=False)

Implements the time evolution of an operator.

Bases: [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate)

Given a fermionic `operator` $H$ and an evolution `time` $t$, this gate
implements the unitary

$$
U = e^{-i t H}.
$$

For $U$ to be unitary, $H$ must be Hermitian. This is the caller’s responsibility
and is not verified.

#### NOTE
Breaking this gate down in fermionic space (into a circuit of smaller [`Evolution`](#qiskit_fermions.circuit.library.Evolution)
gates) is **optional**, and it is what the [`synthesis`](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis) attribute governs. The gate can
equally be handed straight to the fermion-to-qubit stage, which maps and synthesizes it whole,
however many terms its `operator` holds. See
[`synthesis`](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis).

When it is decomposed (by [`decompose()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.decompose), for instance)
[`synthesis`](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis) decides how. It defaults to [`FermionicLieTrotter`](qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter), which splits the
evolution group-by-group when the `operator` has
[`groups`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups) assigned (see
[Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation)), and term-by-term otherwise (exact only when the individual factors
mutually commute). The transpilation process may further alter the decomposition;
[`QDriftTrotterization`](qiskit_fermions.transpiler.passes.QDriftTrotterization.html.md#qiskit_fermions.transpiler.passes.QDriftTrotterization), for example, replaces the evolution with a randomized sample of
its terms. The gates it emits are [`atomic`](#qiskit_fermions.circuit.library.Evolution.atomic), so [`synthesis`](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis) no longer applies to
them.

#### NOTE
The state-vector simulation path does not go through [`synthesis`](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis): it exponentiates the
whole `operator` exactly. Simulating this gate therefore incurs no Trotter error, whereas a
decomposed circuit generally does.

Initializing an instance of this gate can be done with the arguments listed below.

* **Parameters:**
  * **num_modes** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of fermionic modes on which this gate acts.
  * **operator** ([*OperatorTrait*](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait)) – the Hermitian operator $H$ under which to time evolve the acted-upon
    fermionic modes.
  * **time** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – the evolution time $t$ entering the exponent of $e^{-i t H}$. A
    negative value evolves backwards in time.
  * **synthesis** ([*FermionicEvolutionSynthesis*](qiskit_fermions.circuit.library.synthesis.FermionicEvolutionSynthesis.html.md#qiskit_fermions.circuit.library.synthesis.FermionicEvolutionSynthesis) *|* *None*) – the fermion-to-fermion synthesis method with which to decompose this gate, when
    it gets decomposed at all. If `None` (the default), a
    [`FermionicLieTrotter`](qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter.html.md#qiskit_fermions.circuit.library.synthesis.FermionicLieTrotter) instance is used.
  * **atomic** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether this gate is a fully decomposed factor which must not be broken down any
    further in fermionic space. See [`atomic`](#qiskit_fermions.circuit.library.Evolution.atomic).

### Attributes

#### atomic

Whether this gate is a factor that must not be decomposed any further.

A [`FermionicEvolutionSynthesis`](qiskit_fermions.circuit.library.synthesis.FermionicEvolutionSynthesis.html.md#qiskit_fermions.circuit.library.synthesis.FermionicEvolutionSynthesis) marks the factors it emits as atomic, because they
are the result of the split it chose to perform: breaking them down again would discard that
choice. The evolution of a single operator term is atomic for the stronger reason that there
is nothing left to split.

An atomic gate has no definition at all, so [`decompose()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.decompose) leaves it in
place rather than expanding it. It still maps and synthesizes normally at the
fermion-to-qubit level, which is where a factor is turned into actual operations.

This is read-only, for the same reason as [`synthesis`](../pydoc/qiskit_fermions.circuit.library.synthesis.html.md#module-qiskit_fermions.circuit.library.synthesis).

#### synthesis

The fermion-to-fermion synthesis method with which this gate is decomposed.

This is read-only: the gate’s definition is built once and then cached, so a synthesis method
assigned after the fact could not take effect. Pass it to the constructor instead.

#### operator

The operator under which to time evolve the acted-upon fermionic modes.

### Methods

#### inverse(annotated=False)

Returns the inverse of this gate, $e^{+i t H}$.

* **Parameters:**
  **annotated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – ignored. The inverse of this gate is another [`Evolution`](#qiskit_fermions.circuit.library.Evolution), so it never
  needs to be an [`AnnotatedOperation`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.AnnotatedOperation).
* **Returns:**
  An [`Evolution`](#qiskit_fermions.circuit.library.Evolution) gate evolving the same operator for the negated time.
* **Return type:**
  [*Evolution*](#qiskit_fermions.circuit.library.Evolution)

### Protocol Methods

#### \_apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

Applies `exp(-i * time * operator)` after relabeling the operator to global modes.

The operator is relabeled onto its global modes, converted to an
[`ffsim.FermionOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.FermionOperator.html#ffsim.FermionOperator) and handed to [`ffsim.linear_operator()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.linear_operator.html#ffsim.linear_operator), then applied to
the vector via [`scipy.sparse.linalg.expm_multiply()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html#scipy.sparse.linalg.expm_multiply). An operator that is not already a
[`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) is first converted through the
[`SupportsFermionOperator`](qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator) protocol, so any operator type of this package can be
evolved.

* **Parameters:**
  * **vec** (*np.ndarray*) – the state vector to act on.
  * **norb** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of spatial orbitals of the *global* state vector.
  * **nelec** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – either a single integer for a spinless system, or a pair of integers storing the
    numbers of spin alpha and spin beta fermions. An integer selects the spinless mode
    interpretation (the operator’s `norb` modes are alpha orbitals); a pair selects the
    spinful `(orb, spin)` interpretation of the operator’s `2 * norb` modes.
  * **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – whether to copy the vector before operating on it.
  * **freg_indices** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *]*) – the absolute (global) mode indices that this gate’s local modes map onto.
    The operator is relabeled from its local modes to these global modes before being
    applied, mirroring the transpiler’s synthesis pass.
* **Returns:**
  The transformed vector.
* **Raises:**
  * **MissingOptionalLibraryError** – if ffsim is not installed.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the operator does not conserve the `(norb, nelec)` sector. Evolving
        under `exp(-i * time * operator)` only yields a unitary when `operator` maps the
        sector to itself, so ffsim rejects an operator that conserves neither particle number
        nor the z-component of spin.
* **Return type:**
  np.ndarray
