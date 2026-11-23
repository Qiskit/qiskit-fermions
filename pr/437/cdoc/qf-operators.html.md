<a id="qf-operators"></a>

# Operator Representations

The C API provides various data structures for representing fermionic operators in different bases.

| [`QfFermionOperator`](qf-ferm-op.html.md#c.QfFermionOperator)               | An operator in the basis of the second-quantization creation<br/>and annihilation operators.   |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [`QfMajoranaOperator`](qf-maj-op.html.md#c.QfMajoranaOperator)             | An operator in the basis of Majorana fermions.                                                 |
| [`QfEdgeVertexOperator`](qf-edge-op.html.md#c.QfEdgeVertexOperator)         | An operator in the basis of the edge and vertex operators.                                     |
| [`QfTransferVertexOperator`](qf-transfer-op.html.md#c.QfTransferVertexOperator) | An operator in the basis of the transfer and vertex<br/>operators.                             |

## Shared helpers

A few functions are not specific to any one representation and so are provided once for all of them.

| [`qf_edges_free()`](#c.qf_edges_free)   | Frees an edge array returned by a non-commutation-edges accessor.   |
|--------------------------------------------------------------------|---------------------------------------------------------------------|

### void qf_edges_free(uint32_t \*edges, uint64_t num_edges)

Frees an edge array returned by a `qf_op_type_non_commutation_edges` call. 

The array is owned by the library rather than the caller, so it must be released through this
function rather than with `free`. One function serves every operator representation, since the
allocation carries no type information.

The array is released by its size, which this function rebuilds from `num_edges`, so the two
must be exactly the pair that one successful call wrote. Any other count is undefined behavior,
and so is freeing the same array twice. A failed call writes neither out-parameter, so start both
from `NULL` and `0` as below: the free is then a no-op whether or not the call succeeded,
rather than releasing whatever a previous call left in them.

```c
uint32_t *edges = NULL;
uint64_t num_edges = 0;
QfExitCode exit = qf_maj_op_non_commutation_edges(op, &edges, &num_edges);
// ... use the edges ...
qf_edges_free(edges, num_edges);
```

 * **Parameters:**
  * **edges** – A pointer to the edge array to free. Passing `NULL` is a no-op. 
  * **num_edges** – The edge count that was written alongside the array, unchanged.
