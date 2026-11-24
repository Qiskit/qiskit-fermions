# Commutator Generators

The [operator representations](qf-operators.html.md#qf-operators) provide efficient functions
for the computation of various commutators. Since these functions work with
generic operator representations (a notion that does not exist in C), they are
explained generically in the following three sections.
The actual functions contained in the C API for the various operator
representations are listed at the bottom of this page.

## Commutator

### OpType qf_op_type_commutator(const OpType \*op_a, const OpType \*op_b)

#### CAUTION
The function signature here is **generic**! A real one will replace
`OpType` with an actual [operator representations](qf-operators.html.md#qf-operators)
and `op_type` with its matching prefix (for example, `ferm_op` for
[`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator)).

The commutator is defined as:

$$
[A, B] = AB - BA
$$

* **Parameters:**
  * **op_a** – A pointer to the operator $A$.
  * **op_b** – A pointer to the operator $B$.
* **Returns:**
  A pointer to the constructed operator commutator.

### Example

```c
QfFermionOperator *op1 = qf_ferm_op_zero();
QkComplex64 coeff1 = {1.0, 0.0};
bool action1[2] = {true, false};
uint32_t modes1[2] = {0, 0};
qf_ferm_op_add_term(op1, 2, action1, modes1, &coeff1);
QfFermionOperator *op2 = qf_ferm_op_zero();
QkComplex64 coeff2 = {2.0, 0.0};
bool action2[2] = {false, true};
uint32_t modes2[2] = {0, 0};
qf_ferm_op_add_term(op2, 2, action2, modes2, &coeff2);

QfFermionOperator *comm = qf_ferm_op_commutator(op1, op2);
```

## Anti-Commutator

### OpType qf_op_type_anti_commutator(const OpType \*op_a, const OpType \*op_b)

#### CAUTION
The function signature here is **generic**! A real one will replace
`OpType` with an actual [operator representations](qf-operators.html.md#qf-operators)
and `op_type` with its matching prefix (for example, `ferm_op` for
[`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator)).

The anti-commutator is defined as:

$$
\{A, B\} = AB + BA
$$

* **Parameters:**
  * **op_a** – A pointer to the operator $A$.
  * **op_b** – A pointer to the operator $B$.
* **Returns:**
  A pointer to the constructed operator anti-commutator.

### Example

```c
QfFermionOperator *op1 = qf_ferm_op_zero();
QkComplex64 coeff1 = {1.0, 0.0};
bool action1[2] = {true, false};
uint32_t modes1[2] = {0, 0};
qf_ferm_op_add_term(op1, 2, action1, modes1, &coeff1);
QfFermionOperator *op2 = qf_ferm_op_zero();
QkComplex64 coeff2 = {2.0, 0.0};
bool action2[2] = {false, true};
uint32_t modes2[2] = {0, 0};
qf_ferm_op_add_term(op2, 2, action2, modes2, &coeff2);

QfFermionOperator *anti_comm = qf_ferm_op_anti_commutator(op1, op2);
```

## Double-Commutator

### OpType qf_op_type_double_commutator(const OpType \*op_a, const OpType \*op_b, const OpType \*op_c, bool sign)

#### CAUTION
The function signature here is **generic**! A real one will replace
`OpType` with an actual [operator representations](qf-operators.html.md#qf-operators)
and `op_type` with its matching prefix (for example, `ferm_op` for
[`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator)).

The double-commutator is defined as follows (see also Chapter 13.6, *Equation of motion
methods*, page 479 of <sup>[1](#id4)</sup>; the `sign=false` case is the symmetric double commutator
$[A, B, C]$ appearing in Equation (2) of <sup>[2](#id5)</sup>):

If `sign` is `false`, it returns

$$
[[A, B], C]/2 + [A, [B, C]]/2 = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

$$

If `sign` is `true`, it returns

$$
\{[A, B], C\}/2 + \{A, [B, C]\}/2 = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

$$

* <a id='id4'>**[1]**</a> R. McWeeny. Methods of Molecular Quantum Mechanics. 2nd Edition, Academic Press, 1992. ISBN 0-12-486552-6.
* <a id='id5'>**[2]**</a> P. J. Ollitrault, A. Miessen, I. Tavernelli. Molecular Quantum Dynamics: A Quantum Computing Perspective. Acc. Chem. Res. 54, 4229-4238 (2021). [https://doi.org/10.1021/acs.accounts.1c00514](https://doi.org/10.1021/acs.accounts.1c00514)
* **Parameters:**
  * **op_a** – A pointer to the operator $A$.
  * **op_b** – A pointer to the operator $B$.
  * **op_c** – A pointer to the operator $C$.
  * **sign** – the nature of the outer (anti-)commutator as per the definition above.
* **Returns:**
  A pointer to the constructed operator anti-commutator.

### Example

```c
QfFermionOperator *op1 = qf_ferm_op_zero();
QkComplex64 coeff1 = {1.0, 0.0};
bool action1[2] = {true, false};
uint32_t modes1[2] = {0, 0};
qf_ferm_op_add_term(op1, 2, action1, modes1, &coeff1);
QfFermionOperator *op2 = qf_ferm_op_zero();
QkComplex64 coeff2 = {2.0, 0.0};
bool action2[2] = {false, true};
uint32_t modes2[2] = {0, 0};
qf_ferm_op_add_term(op2, 2, action2, modes2, &coeff2);
QfFermionOperator *op3 = qf_ferm_op_zero();
qf_ferm_op_add_term(op3, 2, action1, modes1, &coeff1);
QkComplex64 coeff3 = {2.0, 0.5};
qf_ferm_op_add_term(op3, 2, action2, modes2, &coeff3);

QfFermionOperator *double_comm = qf_ferm_op_double_commutator(op1, op2, op3, false);
```

## Functions

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_ferm_op_commutator(const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op_a, const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op_b)

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_ferm_op_anti_commutator(const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op_a, const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op_b)

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_ferm_op_double_commutator(const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op_a, const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op_b, const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op_c, bool sign)

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_maj_op_commutator(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op_a, const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op_b)

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_maj_op_anti_commutator(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op_a, const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op_b)

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_maj_op_double_commutator(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op_a, const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op_b, const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op_c, bool sign)

### [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*qf_edge_op_commutator(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op_a, const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op_b)

### [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*qf_edge_op_anti_commutator(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op_a, const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op_b)

### [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*qf_edge_op_double_commutator(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op_a, const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op_b, const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op_c, bool sign)

### [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*qf_transfer_op_commutator(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op_a, const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op_b)

### [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*qf_transfer_op_anti_commutator(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op_a, const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op_b)

### [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*qf_transfer_op_double_commutator(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op_a, const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op_b, const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op_c, bool sign)
