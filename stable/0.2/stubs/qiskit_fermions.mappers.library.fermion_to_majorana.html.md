# fermion_to_majorana

### fermion_to_majorana(fer_op)

Map a [`FermionOperator`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator) to a [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator).

* **Parameters:**
  **fer_op** – the fermionic operator to map.
* **Returns:**
  The mapped majorana operator.

## Definition

This function implements the simple transformation:

$$
a^\dagger_j \rightarrow \frac{1}{2} (\gamma_j - i \gamma'_j) ~~\text{and}~~
a_j \rightarrow \frac{1}{2} (\gamma_j + i \gamma'_j)
$$

where $a^\dagger_j$ ($a_j$) is the fermionic creation (annihilation) operator
acting on the $j$-th spin-less fermionic mode, and $\gamma_j$/$\gamma'_j$ are
the two Majorana fermion operators. In the case of the [`MajoranaOperator`](qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator) these will be
stored on the even and odd Majorana modes, respectively.

## Usage

```pycon
>>> from qiskit_fermions.mappers.library import fermion_to_majorana
>>> from qiskit_fermions.operators import FermionOperator
>>> fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
>>> maj_op = fermion_to_majorana(fer_op)
>>> print(format(maj_op.normal_ordered().simplify()))
 5.000000e-1 +0.000000e0j * ()
 0.000000e0-5.000000e-1j * (γ'0 γ0)
```
