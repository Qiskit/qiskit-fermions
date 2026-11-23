# Filtering Functions

### void qf_ferm_op_filter_diagonal_terms([QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*op)

Filters out the terms of an operator that are diagonal in the occupation-number basis. 

This removes every term from the provided [`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator) that is a product of
number operators ($a^\dagger_i a_i$). This includes the constant term (a product of zero
number operators), single number operators, as well as higher-order products such as
$n_i n_j = a^\dagger_i a^\dagger_j a_j a_i$.

#### HINT
A major motivation for filtering such terms arises during the time evolution of operators,
where such diagonal terms, do not affect the time evolution besides introducing a global
phase. Consequently, bitstrings sampled from a time evolved state remain unaffected.
Removing them is therefore recommended, for example when preparing an electronic-structure
Hamiltonian for the qDRIFT time evolution used by SqDRIFT, as it avoids an otherwise
unnecessary sampling overhead.

#### CAUTION
The provided operator *must* be normal-ordered! This is an underlying assumption of the
implementation that is *not* being verified! See [`qf_ferm_op_normal_ordered()`](qf-ferm-op.html.md#c.qf_ferm_op_normal_ordered) for how
to get an operator of that form.

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);

QfFermionOperator *normal = qf_ferm_op_normal_ordered(op, NULL);

qf_ferm_op_filter_diagonal_terms(normal);
```

 #### NOTE
The operator is modified *in place*. If it tracks group indices (see
[`qf_ferm_op_get_groups()`](qf-ferm-op.html.md#c.qf_ferm_op_get_groups)), surviving terms retain their relative grouping but the
group indices are reassigned to a contiguous range starting from 0. You must therefore *not*
rely on the specific group index of any term being preserved across a call to this function.

* **Parameters:**
  * **op** – A pointer to the normal-ordered fermionic operator whose terms are to be filtered (modified in place).
