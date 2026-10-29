# anti_commutator

### anti_commutator(op_a, op_b)

Computes the anti-commutator of two operators.

The anti-commutator is defined as:

$$
\{A, B\} = AB + BA
$$

Operators which support this method must implement the [`SupportsCommutators`](qiskit_fermions.protocols.SupportsCommutators.html.md#qiskit_fermions.protocols.SupportsCommutators) protocol.

#### NOTE
Both inputs must be of the same operator type. This will also determine the output type.

```pycon
>>> from qiskit_fermions.operators import FermionOperator
>>> from qiskit_fermions.operators.library import anti_commutator
>>> op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
>>> op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
>>> comm = anti_commutator(op1, op2)
>>> comm = comm.normal_ordered()
>>> canon = comm.simplify()
>>> assert canon == FermionOperator.zero()
```

* **Parameters:**
  * **op_a** (*T*) – the operator $A$ above.
  * **op_b** (*T*) – the operator $B$ above.
* **Returns:**
  The anti-commutator $\{A, B\}$.
* **Return type:**
  *T*
