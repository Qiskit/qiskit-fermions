<a id="ffsim-relationship-explanation"></a>

# How this package relates to ffsim

#### IMPORTANT
The concepts in this guide are only available in the Python API.

[ffsim](https://qiskit-community.github.io/ffsim/) and this package divide the work rather than overlap, along a single line:

| ffsim owns                                                                                                                                                                                                                                                                                                                                                                                             | `qiskit-fermions` owns                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The high-level ansatz *operators* ([`UCJOpSpinBalanced`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCJOpSpinBalanced.html#ffsim.UCJOpSpinBalanced),<br/>[`UCCSDOpRestrictedReal`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.UCCSDOpRestrictedReal.html#ffsim.UCCSDOpRestrictedReal) and their siblings), built from<br/>coupled-cluster amplitudes or a parameter vector | The fermionic *circuit* ([`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)) those operators become, and its<br/>transpilation into a qubit circuit under any fermion-to-qubit encoding |
| Fast simulation in a compact fixed-particle-number space                                                                                                                                                                                                                                                                                                                                               | The mapper framework and the operator data structures the circuit is built from                                                                                                                                                                 |

The practical consequence runs both ways. The ansatz math lives in ffsim, so [`UCJ`](../stubs/qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ) and
[`UCC`](../stubs/qiskit_fermions.circuit.library.UCC.html.md#qiskit_fermions.circuit.library.UCC) take an ffsim operator directly as their input. Conversely, ffsim’s own Qiskit gates
are specific to the Jordan-Wigner transformation, so lowering one of its ansatz operators through a
*different* encoding is what bringing it here buys you.

Simulation is where the two meet, and this guide explains how. ffsim is this package’s simulation
backend, reached through its own protocols rather than a wrapper: the [SKQD guide](skqd.html.md#skqd-getting-started) calls [`ffsim.apply_unitary()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.apply_unitary.html#ffsim.apply_unitary), [`ffsim.linear_operator()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.linear_operator.html#ffsim.linear_operator), and
[`ffsim.sample_state_vector()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.sample_state_vector.html#ffsim.sample_state_vector) directly on [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit)objects and
[`FermionOperator`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)objects, with no conversion step.

## Plugging into ffsim’s simulation interface

[ffsim](https://qiskit-community.github.io/ffsim/) is a high-performance simulator for fermionic quantum circuits that exploits
particle number and spin-Z conservation to represent state vectors more compactly than a
generic $2^n$-dimensional qubit statevector. It defines protocols that any object
can implement to participate in its simulation machinery (see [`qiskit_fermions.protocols`](../pydoc/qiskit_fermions.protocols.html.md#module-qiskit_fermions.protocols)
for this package’s own protocols, which follow the same design):

- [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary), by the method `_apply_unitary_(vec, norb, nelec, copy)`,
  applying the object as a unitary to a fixed-particle-number state vector
- [`ffsim.SupportsLinearOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsLinearOperator.html#ffsim.SupportsLinearOperator), by the method `_linear_operator_(norb, nelec)`, returning a
  [`scipy.sparse.linalg.LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy.sparse.linalg.LinearOperator) view of the object on that same sector
- [`ffsim.SupportsTrace`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsTrace.html#ffsim.SupportsTrace), by the method `_trace_(norb, nelec)`, returning the object’s trace
  on that sector.

This package implements that interface: every fermionic gate in
[`qiskit_fermions.circuit.library`](../pydoc/qiskit_fermions.circuit.library.html.md#module-qiskit_fermions.circuit.library) provides `_apply_unitary_`, and [`FermionOperator`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)
provides `_linear_operator_` and `_trace_`. That is what makes this package’s operators and
circuits *compatible* with ffsim, so its tools work on them natively, with no conversion step.
[`ffsim.apply_unitary()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.apply_unitary.html#ffsim.apply_unitary)
can simulate a [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) end to end, [`ffsim.linear_operator()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.linear_operator.html#ffsim.linear_operator) can diagonalize a
[`FermionOperator`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator), and sampling utilities like [`ffsim.sample_state_vector()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.sample_state_vector.html#ffsim.sample_state_vector) (used in the
[SKQD guide](skqd.html.md#skqd-getting-started) to turn a simulated statevector into measurement counts)
work without modification.

What `_linear_operator_` returns is an ordinary SciPy
[`LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy.sparse.linalg.LinearOperator), so [`scipy.sparse.linalg.eigsh()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh) and friends work
on it too.

```pycon
>>> import ffsim
>>> import numpy as np
>>>
>>> from qiskit_fermions.circuit import FermionicCircuit
>>> from qiskit_fermions.circuit.library import Evolution
>>> from qiskit_fermions.operators import FermionOperator, ann, cre
>>>
>>> norb, nelec = 2, (1, 1)
>>> hamiltonian = FermionOperator.from_terms([
...     ([cre(0), ann(1)], 0.5),
...     ([cre(1), ann(0)], 0.5),
... ])
>>>
>>> circuit = FermionicCircuit(2 * norb)
>>> circuit.append(Evolution(2 * norb, hamiltonian, time=1.0), circuit.modes)
>>>
>>> reference = ffsim.hartree_fock_state(norb, nelec)
>>> state = ffsim.apply_unitary(reference, circuit, norb=norb, nelec=nelec)  # native ffsim call
```

Coupling through ffsim’s protocols is what buys this: its actively developed ecosystem of simulation
and sampling utilities applies to this package’s circuits and operators unchanged, and users already
working with ffsim can mix in this package’s gates and operators without learning another simulation
API.

## ffsim is an optional dependency

ffsim is declared as an optional extra (`pip install "qiskit-fermions[ffsim]"`, or transitively by
`[all]`), guarded at runtime by [`HAS_FFSIM`](../pydoc/qiskit_fermions.utils.optionals.html.md#qiskit_fermions.utils.optionals.HAS_FFSIM) (as was shown
above). Installing it unlocks simulation: [`ffsim.SupportsLinearOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsLinearOperator.html#ffsim.SupportsLinearOperator) and
[`ffsim.SupportsTrace`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsTrace.html#ffsim.SupportsTrace) are implemented by converting this package’s [`FermionOperator`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) into
an [`ffsim.FermionOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.FermionOperator.html#ffsim.FermionOperator), so without ffsim they raise
[`MissingOptionalLibraryError`](https://quantum.cloud.ibm.com/docs/api/qiskit/exceptions#qiskit.exceptions.MissingOptionalLibraryError). Everything that does not simulate (building
operators, mapping them, and transpiling the resulting circuits) needs none of it.

[ffsim](https://qiskit-community.github.io/ffsim/) transitively depends on [PySCF](https://pyscf.org/), which does not support Windows, so
on Windows the extra resolves to nothing (a silent no-op through a `sys_platform` marker) rather
than an install failure. Windows users who want to simulate can do so through the [Windows
Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) in the meantime.

## Fermionic simulation lives in a fixed particle-number sector

Both `_apply_unitary_` and `_linear_operator_` take an `nelec` argument and represent the state
vector over the *fixed-particle-number determinant basis* for that `(norb, nelec)` sector,
mirroring ffsim’s (and, transitively, PySCF’s) FCI space setup, rather than the full
$2^{\text{num\_modes}}$-dimensional space a general qubit simulator would use. This is a much
smaller space (its size is a product of binomial coefficients rather than a power of two), but it
comes with a hard restriction. Only operators and gates that preserve particle number (and, in
the spinful case, each spin species’ particle number individually) can be represented in it. An
operator whose action would move amplitude to a different particle number has nowhere to put it.

ffsim resolves this by rejecting such an operator outright: converting one to a linear operator
raises a [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError) naming which conservation law fails. That is the right default, because
applying $\exp(-i t H)$ for a Hamiltonian $H$ with a non-particle-conserving term would
otherwise turn a would-be unitary into a non-unitary, physically-meaningless map.

- [`Evolution`](../stubs/qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) surfaces that rejection from the operator it evolves.
- [`OrbitalRotation`](../stubs/qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) checks that its (embedded) rotation matrix is block-diagonal across the
  alpha/beta split in the spinful case, and raises a [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError) if it mixes spin sectors.

```pycon
>>> from qiskit_fermions.circuit.library import Evolution
>>>
>>> non_conserving = FermionOperator.from_terms([([cre(0), cre(1)], 1.0)])  # creates 2 particles
>>> gate = Evolution(2 * norb, non_conserving, time=1.0)
>>>
>>> try:
...     gate._apply_unitary_(reference, norb, nelec, copy=True)
... except ValueError as exc:
...     print("rejected:", exc)
rejected: The given FermionOperator could not be converted to a LinearOperator because it does not conserve particle number and the z component of spin. Conserves particle number: False Conserves spin z: False
```

Non-particle-preserving simulation is therefore out of reach in fermionic space, by construction of
the fixed-sector representation, and that is the price of the compact FCI space which makes
fermionic simulation tractable in the first place. If your algorithm needs particle-number-violating
operators (for example, a qubit-native error channel, or an operator built for a mapped Hamiltonian
that does not conserve particle numbers term by term), transpile to qubits first and simulate the
resulting [`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit) with a qubit-level simulator instead. Any
transpilation route works for this; the
[fermionic circuit](circuit.html.md#fermionic-circuit-explanation) and [transpilation](transpilation.html.md#transpilation-explanation) guides go into more detail, but
[`generate_preset_jw_pass_manager()`](../stubs/qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager.html.md#qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager) is a reasonable default
choice.

## The block-spin convention for spinful systems

`nelec` is typed `int | tuple[int, int]`, and its type, not a separate flag, is what selects
one of the supported mode layouts:

- An **int** selects the **spinless** interpretation. The `norb` modes are treated directly as
  spinless orbitals, and `nelec` is the total particle count.
- A **pair** `(n_alpha, n_beta)` selects the **spinful** interpretation of `2 * norb` modes under
  a fixed **block-spin convention**. Modes `0 .. norb` are the alpha (spin-up) orbitals and modes
  `norb .. 2 * norb` are the beta (spin-down) orbitals. Each spin species is conserved
  *independently*; an alpha-only term can move an electron between alpha modes but never into a
  beta mode, and vice versa.

This dispatch happens at every ffsim-protocol entry point in this package (gate constructors,
`_apply_unitary_placed_` implementations, and the native Rust kernel’s sector compilation),
and it is the convention used throughout the [LUCJ](lucj.html.md#lucj-transpilation) and
[SKQD](skqd.html.md#skqd-getting-started) guides (for example,
[`InitializeModes.from_hartree_fock()`](../stubs/qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes.from_hartree_fock) fills alpha occupations into modes `0 .. n_alpha` and
beta occupations into modes `norb .. norb + n_beta`). It is worth contrasting this with the
[`qiskit_fermions.operators`](../pydoc/qiskit_fermions.operators.html.md#module-qiskit_fermions.operators) module and [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) in general, which use
*generic* mode indices with no inherent spin semantics (see the [fermionic circuit guide](circuit.html.md#fermionic-circuit-explanation)). The block-spin meaning is imposed only when a spinful `nelec`
is supplied to a simulation call, not baked into the operator or circuit representation itself.

```pycon
>>> from qiskit_fermions.circuit.library import InitializeModes
>>>
>>> norb, nelec = 3, (2, 1)
>>> init = InitializeModes.from_hartree_fock(norb, nelec)
>>> print([bool(occ) for occ in init.occupation])  # alpha modes 0,1; beta mode 3 (= norb + 0)
[True, True, False, True, False, False]
```

### Next steps

- Walk through the [SKQD](skqd.html.md#skqd-getting-started) guide to see [`ffsim.apply_unitary()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.apply_unitary.html#ffsim.apply_unitary) and
  [`ffsim.sample_state_vector()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.sample_state_vector.html#ffsim.sample_state_vector) used together to sample circuits built from this package’s gates,
  and to evaluate a Hamiltonian expectation value through [`ffsim.linear_operator()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.linear_operator.html#ffsim.linear_operator) (itself backed
  by the same `_linear_operator_` protocol method described here).
- Walk through the [LUCJ](lucj.html.md#lucj-transpilation) guide for the other half of the story: taking an
  ffsim ansatz operator through this package’s transpilation pipeline and onto a device coupling map.
- Read the [fermionic circuit guide](circuit.html.md#fermionic-circuit-explanation) for the generic,
  spin-agnostic mode indexing used outside of simulation calls.
- Read the [transpilation guide](transpilation.html.md#transpilation-explanation) for how to leave fermionic space
  : and simulate on qubits, which is required for non-particle-conserving operators.
- Browse [`qiskit_fermions.protocols`](../pydoc/qiskit_fermions.protocols.html.md#module-qiskit_fermions.protocols) for the full catalog of protocols this package defines,
  including the conversion protocols ([`SupportsFermionOperator`](../stubs/qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator),
  [`SupportsMajoranaOperator`](../stubs/qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator)) that are unrelated to ffsim.
- Read [ffsim’s own guides](https://qiskit-community.github.io/ffsim/how-to-guides/) for the
  workflows it owns, in particular
  [building a UCJ ansatz](https://qiskit-community.github.io/ffsim/how-to-guides/qiskit-lucj.html) and
  [optimizing one variationally](https://qiskit-community.github.io/ffsim/how-to-guides/simulate-vqe.html). The parameter
  vectors those guides drive are what [`UCJ`](../stubs/qiskit_fermions.circuit.library.UCJ.html.md#qiskit_fermions.circuit.library.UCJ) and [`UCC`](../stubs/qiskit_fermions.circuit.library.UCC.html.md#qiskit_fermions.circuit.library.UCC) accept, through the operators
  they build.
