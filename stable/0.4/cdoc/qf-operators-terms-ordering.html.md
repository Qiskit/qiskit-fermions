# Ordering Functions

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_ferm_op_canonical_order(const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op)

Returns a copy of a fermionic operator with its terms in a canonical order. 

The terms are sorted into a canonical order that depends only on each term’s structure (the
operator string it represents) and not on its coefficient. The order is therefore deterministic
for a given set of terms regardless of how the operator was assembled. The terms themselves are
left untouched — this only reorders them, it does not simplify or normal-order the operator.

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[2] = {true, false};
uint32_t modes_a[2] = {1, 0};
QkComplex64 coeff_a = {1.0, 0.0};
qf_ferm_op_add_term(op, 2, actions, modes_a, &coeff_a);
uint32_t modes_b[2] = {0, 1};
QkComplex64 coeff_b = {2.0, 0.0};
qf_ferm_op_add_term(op, 2, actions, modes_b, &coeff_b);

QfFermionOperator *ordered = qf_ferm_op_canonical_order(op);
```

 #### NOTE
Any group indices (see [`qf_ferm_op_get_groups()`](qf-ferm-op.html.md#c.qf_ferm_op_get_groups)) are preserved: each term carries its
group index along as it moves, since a group index is a per-term tag independent of term
order. Terms of the same group are simply no longer contiguous afterwards. If the input has
no groups, neither does the result.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_maj_op_canonical_order(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op)

Returns a copy of a Majorana operator with its terms in a canonical order. 

The terms are sorted into a canonical order that depends only on each term’s structure (the
operator string it represents) and not on its coefficient. The order is therefore deterministic
for a given set of terms regardless of how the operator was assembled. The terms themselves are
left untouched — this only reorders them, it does not simplify or normal-order the operator.

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes_a[2] = {1, 0};
QkComplex64 coeff_a = {1.0, 0.0};
qf_maj_op_add_term(op, 2, modes_a, &coeff_a);
uint32_t modes_b[2] = {0, 1};
QkComplex64 coeff_b = {2.0, 0.0};
qf_maj_op_add_term(op, 2, modes_b, &coeff_b);

QfMajoranaOperator *ordered = qf_maj_op_canonical_order(op);
```

 #### NOTE
Any group indices (see [`qf_maj_op_get_groups()`](qf-maj-op.html.md#c.qf_maj_op_get_groups)) are preserved: each term carries its
group index along as it moves, since a group index is a per-term tag independent of term
order. Terms of the same group are simply no longer contiguous afterwards. If the input has
no groups, neither does the result.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.

### [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*qf_edge_op_canonical_order(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op)

Returns a copy of an edge-vertex operator with its terms in a canonical order. 

The terms are sorted into a canonical order that depends only on each term’s structure (the
operator string it represents) and not on its coefficient. The order is therefore deterministic
for a given set of terms regardless of how the operator was assembled. The terms themselves are
left untouched — this only reorders them, it does not simplify or normal-order the operator.

```c
QfEdgeVertexOperator *op = qf_edge_op_zero();
uint32_t left_a[1] = {1};
uint32_t right_a[1] = {2};
QkComplex64 coeff_a = {1.0, 0.0};
qf_edge_op_add_term(op, 1, left_a, right_a, &coeff_a);
uint32_t left_b[1] = {0};
uint32_t right_b[1] = {1};
QkComplex64 coeff_b = {2.0, 0.0};
qf_edge_op_add_term(op, 1, left_b, right_b, &coeff_b);

QfEdgeVertexOperator *ordered = qf_edge_op_canonical_order(op);
```

 #### NOTE
Any group indices (see [`qf_edge_op_get_groups()`](qf-edge-op.html.md#c.qf_edge_op_get_groups)) are preserved: each term carries its
group index along as it moves, since a group index is a per-term tag independent of term
order. Terms of the same group are simply no longer contiguous afterwards. If the input has
no groups, neither does the result.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.

### [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*qf_transfer_op_canonical_order(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op)

Returns a copy of a transfer-vertex operator with its terms in a canonical order. 

The terms are sorted into a canonical order that depends only on each term’s structure (the
operator string it represents) and not on its coefficient. The order is therefore deterministic
for a given set of terms regardless of how the operator was assembled. The terms themselves are
left untouched — this only reorders them, it does not simplify or normal-order the operator.

```c
QfTransferVertexOperator *op = qf_transfer_op_zero();
uint32_t left_a[1] = {1};
uint32_t right_a[1] = {2};
QkComplex64 coeff_a = {1.0, 0.0};
qf_transfer_op_add_term(op, 1, left_a, right_a, &coeff_a);
uint32_t left_b[1] = {0};
uint32_t right_b[1] = {1};
QkComplex64 coeff_b = {2.0, 0.0};
qf_transfer_op_add_term(op, 1, left_b, right_b, &coeff_b);

QfTransferVertexOperator *ordered = qf_transfer_op_canonical_order(op);
```

 #### NOTE
Any group indices (see [`qf_transfer_op_get_groups()`](qf-transfer-op.html.md#c.qf_transfer_op_get_groups)) are preserved: each term carries
its group index along as it moves, since a group index is a per-term tag independent of term
order. Terms of the same group are simply no longer contiguous afterwards. If the input has
no groups, neither does the result.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_ferm_op_group_order(const [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op)

Returns a copy of a fermionic operator with its terms ordered by group index. 

The terms are sorted by their group index alone, which makes each group one contiguous run of
terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
relative order. The terms themselves are left untouched — this only reorders them, it does not
simplify or normal-order the operator.

Group indices say only which terms belong together, so this changes the operator’s *layout*, not
its value. What the layout buys is lookup cost:
[`qf_ferm_op_split_out_groups()`](qf-ferm-op.html.md#c.qf_ferm_op_split_out_groups) has to scan every term to find the requested groups in
general, but on a group-ordered operator each group’s terms form one range it can locate by binary
search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.

```c
QfFermionOperator *op = qf_ferm_op_zero();
bool actions[2] = {true, false};
uint32_t modes_a[2] = {0, 1};
QkComplex64 coeff_a = {1.0, 0.0};
qf_ferm_op_add_term(op, 2, actions, modes_a, &coeff_a);
uint32_t modes_b[2] = {1, 0};
QkComplex64 coeff_b = {2.0, 0.0};
qf_ferm_op_add_term(op, 2, actions, modes_b, &coeff_b);
uint32_t groups[2] = {1, 0};
qf_ferm_op_set_groups(op, groups, 2);

QfFermionOperator *ordered = qf_ferm_op_group_order(op);
```

 #### NOTE
An operator tracking no group indices (see [`qf_ferm_op_get_groups()`](qf-ferm-op.html.md#c.qf_ferm_op_get_groups)) has nothing to
order by and is returned as an unchanged copy. The result tracks no groups either.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.

### [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*qf_maj_op_group_order(const [QfMajoranaOperator](qf-maj-op.html.md#c.QfMajoranaOperator) \*op)

Returns a copy of a Majorana operator with its terms ordered by group index. 

The terms are sorted by their group index alone, which makes each group one contiguous run of
terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
relative order. The terms themselves are left untouched — this only reorders them, it does not
simplify or normal-order the operator.

Group indices say only which terms belong together, so this changes the operator’s *layout*, not
its value. What the layout buys is lookup cost:
[`qf_maj_op_split_out_groups()`](qf-maj-op.html.md#c.qf_maj_op_split_out_groups) has to scan every term to find the requested groups in
general, but on a group-ordered operator each group’s terms form one range it can locate by binary
search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.

```c
QfMajoranaOperator *op = qf_maj_op_zero();
uint32_t modes_a[2] = {0, 1};
QkComplex64 coeff_a = {1.0, 0.0};
qf_maj_op_add_term(op, 2, modes_a, &coeff_a);
uint32_t modes_b[2] = {2, 3};
QkComplex64 coeff_b = {2.0, 0.0};
qf_maj_op_add_term(op, 2, modes_b, &coeff_b);
uint32_t groups[2] = {1, 0};
qf_maj_op_set_groups(op, groups, 2);

QfMajoranaOperator *ordered = qf_maj_op_group_order(op);
```

 #### NOTE
An operator tracking no group indices (see [`qf_maj_op_get_groups()`](qf-maj-op.html.md#c.qf_maj_op_get_groups)) has nothing to
order by and is returned as an unchanged copy. The result tracks no groups either.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.

### [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*qf_edge_op_group_order(const [QfEdgeVertexOperator](qf-edge-op.html.md#c.QfEdgeVertexOperator) \*op)

Returns a copy of an edge-vertex operator with its terms ordered by group index. 

The terms are sorted by their group index alone, which makes each group one contiguous run of
terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
relative order. The terms themselves are left untouched — this only reorders them, it does not
simplify or normal-order the operator.

Group indices say only which terms belong together, so this changes the operator’s *layout*, not
its value. What the layout buys is lookup cost:
[`qf_edge_op_split_out_groups()`](qf-edge-op.html.md#c.qf_edge_op_split_out_groups) has to scan every term to find the requested groups in
general, but on a group-ordered operator each group’s terms form one range it can locate by binary
search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.

```c
QfEdgeVertexOperator *op = qf_edge_op_zero();
uint32_t left_a[1] = {0};
uint32_t right_a[1] = {1};
QkComplex64 coeff_a = {1.0, 0.0};
qf_edge_op_add_term(op, 1, left_a, right_a, &coeff_a);
uint32_t left_b[1] = {1};
uint32_t right_b[1] = {2};
QkComplex64 coeff_b = {2.0, 0.0};
qf_edge_op_add_term(op, 1, left_b, right_b, &coeff_b);
uint32_t groups[2] = {1, 0};
qf_edge_op_set_groups(op, groups, 2);

QfEdgeVertexOperator *ordered = qf_edge_op_group_order(op);
```

 #### NOTE
An operator tracking no group indices (see [`qf_edge_op_get_groups()`](qf-edge-op.html.md#c.qf_edge_op_get_groups)) has nothing to
order by and is returned as an unchanged copy. The result tracks no groups either.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.

### [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*qf_transfer_op_group_order(const [QfTransferVertexOperator](qf-transfer-op.html.md#c.QfTransferVertexOperator) \*op)

Returns a copy of a transfer-vertex operator with its terms ordered by group index. 

The terms are sorted by their group index alone, which makes each group one contiguous run of
terms and the group indices non-decreasing. The sort is stable, so terms within a group keep their
relative order. The terms themselves are left untouched — this only reorders them, it does not
simplify or normal-order the operator.

Group indices say only which terms belong together, so this changes the operator’s *layout*, not
its value. What the layout buys is lookup cost:
[`qf_transfer_op_split_out_groups()`](qf-transfer-op.html.md#c.qf_transfer_op_split_out_groups) has to scan every term to find the requested groups in
general, but on a group-ordered operator each group’s terms form one range it can locate by binary
search, so a lookup costs what the *requested* groups cost rather than what the *held* terms cost.

```c
QfTransferVertexOperator *op = qf_transfer_op_zero();
uint32_t left_a[1] = {0};
uint32_t right_a[1] = {1};
QkComplex64 coeff_a = {1.0, 0.0};
qf_transfer_op_add_term(op, 1, left_a, right_a, &coeff_a);
uint32_t left_b[1] = {1};
uint32_t right_b[1] = {2};
QkComplex64 coeff_b = {2.0, 0.0};
qf_transfer_op_add_term(op, 1, left_b, right_b, &coeff_b);
uint32_t groups[2] = {1, 0};
qf_transfer_op_set_groups(op, groups, 2);

QfTransferVertexOperator *ordered = qf_transfer_op_group_order(op);
```

 #### NOTE
An operator tracking no group indices (see [`qf_transfer_op_get_groups()`](qf-transfer-op.html.md#c.qf_transfer_op_get_groups)) has nothing to
order by and is returned as an unchanged copy. The result tracks no groups either.

* **Parameters:**
  * **op** – A pointer to the operator whose terms are to be reordered.
* **Returns:**
  A pointer to the created operator.
