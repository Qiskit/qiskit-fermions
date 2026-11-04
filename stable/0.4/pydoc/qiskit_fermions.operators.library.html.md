<a id="module-qiskit_fermions.operators.library"></a>

# Operator Library

Building on top of the operator representations provided by [`operators`](qiskit_fermions.operators.html.md#module-qiskit_fermions.operators), this
module provides various tools to simplify the construction of commonly used operators.

## Commutator Generators

Operator classes can implement the [`SupportsCommutators`](../stubs/qiskit_fermions.protocols.SupportsCommutators.html.md#qiskit_fermions.protocols.SupportsCommutators) [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol), which
allows them to be used with the functions listed in the table below, to quickly compute multiple
commutator variants.

## Common Operators

Various common operators can easily be generated from constructor methods.
This section provides an overview of these methods for a quick reference, grouped by category.

### Electronic Integrals

The constructor methods listed here generally take the coefficients of electronic structure
Hamiltonians as an input. Different flavors exist:

* `tril`: these methods consume 1-dimensional arrays of flattened (generalized) triangular indices
* `full`: these methods consume high-dimensional arrays
* `spin`: these methods take separate arrays for the different spin species
* `sym`: these methods take a single array for one spin species and infer the other spin species

The same `tril` and `full` vocabulary names the layout of arrays that are *returned* rather than
consumed, such as the integral accessors of [`FCIDump`](../stubs/qiskit_fermions.operators.library.FCIDump.html.md#qiskit_fermions.operators.library.FCIDump).

* 1-Body Terms

| [`FermionOperator.from_1body_tril_spin_sym()`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator.from_1body_tril_spin_sym)   | Constructs from spin-symmetric triangular 1-body integrals.   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`FermionOperator.from_1body_tril_spin()`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator.from_1body_tril_spin)           | Constructs from separate spin triangular 1-body integrals.    |
* 2-Body Terms

| [`FermionOperator.from_2body_tril_spin_sym()`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator.from_2body_tril_spin_sym)   | Constructs from spin-symmetric triangular 2-body integrals.   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`FermionOperator.from_2body_tril_spin()`](../stubs/qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator.from_2body_tril_spin)           | Constructs from separate spin triangular 2-body integrals.    |

## Other Generators

Finally, the following additional operator generator utilities exist in this module:
