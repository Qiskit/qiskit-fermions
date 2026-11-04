<a id="module-qiskit_fermions.operators.terms"></a>

# Operator Terms

This module collects routines that operate on the individual terms of an operator, partitioning
them based on their structure. The submodules below group related functionality.

## Grouping

Please refer to [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation) for a detailed explanation of this module’s
functionality.

### Assignment

Rather than always relying on the user to provide the group indices themselves, this module provides
a collection of functions which determine the grouping information automatically.

### Analysis

Group indices carry no intrinsic meaning: the array only says which terms belong together, never
why. Accordingly, none of the functions below reports whether a grouping is “correct”: each
answers one narrow, stated question about an existing grouping, leaving the interpretation to you.
They are provided so that an assumption a downstream consumer makes about a grouping can be checked
up front, rather than being paid for on every call.

## Filtering

This module provides convenience functions for removing terms from an operator that do not
contribute meaningfully to a downstream computation.

### Members

## Ordering

This module provides functions to reorder the terms of an operator. While the
resulting operators are mathematically identical, the term order can have
significant implications on algorithm behavior at runtime, due to the term
iteration order.

Group indices are a per-term tag that says nothing about where those terms sit, so the terms of one
group are in general scattered throughout an operator. [`group_order()`](../stubs/qiskit_fermions.operators.terms.group_order.html.md#qiskit_fermions.operators.terms.group_order) gathers each group into one
contiguous run, which is what lets
[`split_out_groups()`](../stubs/qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.split_out_groups) locate a requested group by binary
search instead of scanning every term.

#### NOTE
Unlike [`canonical_order()`](../stubs/qiskit_fermions.operators.terms.canonical_order.html.md#qiskit_fermions.operators.terms.canonical_order), [`order_terms()`](../stubs/qiskit_fermions.operators.terms.order_terms.html.md#qiskit_fermions.operators.terms.order_terms) has no counterpart in the C API.

### Members
