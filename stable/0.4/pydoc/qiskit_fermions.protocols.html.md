<a id="module-qiskit_fermions.protocols"></a>

# Protocols

This module provides various protocols used throughout the package.

A [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol) is an interface that a class can fulfill structurally, without
explicitly inheriting from it. For example, any class that implements a `__len__` method
satisfies [`Sized`](https://docs.python.org/3/library/typing.html#typing.Sized), whether or not it says so anywhere in its class hierarchy.

The protocols in this module follow the same idea, but each one centers on a single, otherwise
unremarkable dunder-style method (for example `_majorana_operator_`). A class implementing that
one method thereby satisfies the corresponding protocol ([`SupportsMajoranaOperator`](../stubs/qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator) in this
example) and becomes usable wherever that protocol is expected, without any explicit registration.
Note that “protocol” is sometimes also used loosely to refer to a type-agnostic dispatch function
built on top of a protocol (for example [`fermion_operator()`](../stubs/qiskit_fermions.mappers.library.fermion_operator.html.md#qiskit_fermions.mappers.library.fermion_operator)) rather than the interface itself;
the distinction should be clear from context.

Some protocols below are paired with a small, type-agnostic **helper function** that dispatches to
the protocol method on whatever object it is given.

| Protocol                                                                                                                         | Method                                                                           | Helper function                                                                                                                                                                                                                                                                                                                                            |
|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`SupportsFermionOperator`](../stubs/qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator)       | `_fermion_operator_(self)`                                                       | [`fermion_operator()`](../stubs/qiskit_fermions.mappers.library.fermion_operator.html.md#qiskit_fermions.mappers.library.fermion_operator)                                                                                                                                                                                                                                       |
| [`SupportsMajoranaOperator`](../stubs/qiskit_fermions.protocols.SupportsMajoranaOperator.html.md#qiskit_fermions.protocols.SupportsMajoranaOperator)     | `_majorana_operator_(self)`                                                      | [`majorana_operator()`](../stubs/qiskit_fermions.mappers.library.majorana_operator.html.md#qiskit_fermions.mappers.library.majorana_operator)                                                                                                                                                                                                                                     |
| [`SupportsApplyUnitaryPlaced`](../stubs/qiskit_fermions.protocols.SupportsApplyUnitaryPlaced.html.md#qiskit_fermions.protocols.SupportsApplyUnitaryPlaced) | `_apply_unitary_placed_(self, vec, norb, nelec, copy, freg_indices)`             | –                                                                                                                                                                                                                                                                                                                                                          |
| [`SupportsCommutators`](../stubs/qiskit_fermions.protocols.SupportsCommutators.html.md#qiskit_fermions.protocols.SupportsCommutators)               | `_commutator_`, `_anti_commutator_`, `_double_commutator_` (all `@staticmethod`) | [`commutator()`](../stubs/qiskit_fermions.operators.library.commutator.html.md#qiskit_fermions.operators.library.commutator), [`anti_commutator()`](../stubs/qiskit_fermions.operators.library.anti_commutator.html.md#qiskit_fermions.operators.library.anti_commutator), [`double_commutator()`](../stubs/qiskit_fermions.operators.library.double_commutator.html.md#qiskit_fermions.operators.library.double_commutator) |

## Protocols implemented from ffsim

Simulation is [ffsim](https://qiskit-community.github.io/ffsim/)’s concern, and its protocols are
defined there rather than restated here. The gates and operators of this package implement the
following, so ffsim’s dispatchers accept them directly:

| ffsim protocol                                                                                                                                      | Method                                          | Implemented by                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ffsim.SupportsApplyUnitary`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsApplyUnitary.html#ffsim.SupportsApplyUnitary)       | `_apply_unitary_(self, vec, norb, nelec, copy)` | [`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) and every [`FermionicGate`](../stubs/qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) |
| [`ffsim.SupportsLinearOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsLinearOperator.html#ffsim.SupportsLinearOperator) | `_linear_operator_(self, norb, nelec)`          | [`FermionOperator`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)                                                                                                                |
| [`ffsim.SupportsTrace`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsTrace.html#ffsim.SupportsTrace)                            | `_trace_(self, norb, nelec)`                    | [`FermionOperator`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator)                                                                                                                |

#### CAUTION
[`ffsim.SupportsFermionOperator`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.SupportsFermionOperator.html#ffsim.SupportsFermionOperator) shares the `_fermion_operator_` method name with
this package’s [`SupportsFermionOperator`](../stubs/qiskit_fermions.protocols.SupportsFermionOperator.html.md#qiskit_fermions.protocols.SupportsFermionOperator), but the two are *not* the same contract: ours
returns this package’s [`FermionOperator`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) rather than ffsim’s unrelated type. Since
[`ffsim.fermion_operator()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.fermion_operator.html#ffsim.fermion_operator) dispatches purely on the method name, calling it on an object of
this package returns our type.
