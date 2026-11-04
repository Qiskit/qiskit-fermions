# FermionicGivensDecomposition

### *class* FermionicGivensDecomposition(, atol=1e-12)

The decomposition of an orbital rotation into Givens rotations and phases.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The unitary $U$ of the [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) is factorized with
[`givens_decomposition()`](qiskit_fermions.linalg.givens_decomposition.html.md#qiskit_fermions.linalg.givens_decomposition) into at most $n(n-1)/2$ rotations between *adjacent* modes
plus $n$ single-mode phases, the rectangular mesh of Clements et al. <sup>[1](#id2)</sup>. Every factor
$F$ is itself an orbital rotation, whose many-body unitary is $e^{-i H_F}$ for the
Hermitian generator $H_F = \sum_{ij} i \log(F)_{ij} \, a^\dagger_i a_j$. It is therefore
emitted as an [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate at unit time on the one or two modes it acts on.

The emitted gates are `atomic`, and the operator of each holds a single group.
The terms of a two-mode generator, $a^\dagger_0 a_1$ and $a^\dagger_1 a_0$, do not
commute with each other, so splitting it up any further would only approximate the factor: the
fermion-to-qubit stage maps and synthesizes it whole. Under Jordan-Wigner, the generator of a
rotation between adjacent modes is a hop between adjacent qubits, which the qubit stage can
recognize (see [`HopEvolutionSynthesis`](qiskit_fermions.transpiler.passes.synthesis.HopEvolutionSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.HopEvolutionSynthesis)).

Because $U \mapsto \exp(\sum_{ij} \log(U)_{ij} a^\dagger_i a_j)$ is a group homomorphism,
the product of the factors equals the full rotation exactly. The decomposition itself therefore
adds no error: only the synthesis of the individual factors can.

* <a id='id2'>**[1]**</a> W. R. Clements et al., Optimal design for universal multiport interferometers, Optica 3, 1460 (2016), [arXiv:1603.08788](https://arxiv.org/abs/1603.08788).

Initializing an instance of this synthesis method can be done with the arguments below.

* **Parameters:**
  **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – the absolute tolerance below which factors and entries are dropped. This is
  applied to the entries of the generator of a factor and to the angle of a
  single-mode phase.

### Attributes

#### atol

The magnitude below which entries of a generator and angles of a phase are dropped.

### Methods

#### synthesize(gate)

Synthesizes the gate into the mesh of atomic [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates.

See the class documentation for the decomposition this implements.

* **Parameters:**
  **gate** ([*OrbitalRotation*](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation)) – the gate to synthesize.
* **Returns:**
  A [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) holding one [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate per two-mode rotation
  and per non-trivial phase, in the order in which they are applied. The rotations come
  first, then the phases.
* **Return type:**
  [FermionicCircuit](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)
