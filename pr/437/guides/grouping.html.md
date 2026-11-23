<a id="grouping-explanation"></a>

# Group operator terms: use the operator structure

Grouping is one of the operations under [`qiskit_fermions.operators.terms`](../pydoc/qiskit_fermions.operators.terms.html.md#module-qiskit_fermions.operators.terms),
which collects routines that partition the terms of an operator based on their
structure.

As described in the [operators guide](operators.html.md#term-grouping), operators can store a
**groups array**, an optional part of the sparse data structure that associates each
term with a group index. Terms sharing the same group index form a group. This lets you use systematically use of the structure in downstream processing.

By grouping related terms (whether by physical properties, algebraic relationships,
or problem-specific symmetries), you can unlock several benefits:

- **Optimized circuit synthesis**: Grouped terms carry problem-specific
  information through the stack, enabling simpler optimization of the
  synthesized circuits. This results in reduced circuit depth and gate count.
- **Physical structure preservation**: Groups encode meaningful structure from the
  original problem (for example, interaction patterns or symmetries) that can be used
  for more physically meaningful decompositions.
- **Flow set selection**: Grouping can naturally encode flow sets and other graph-theoretic
  structures relevant to quantum simulation, as shown in <sup>[1](#id4)</sup>.
- **Improved algorithmic stability**: As demonstrated in the [SqDRIFT routine](sqdrift.html.md#sqdrift-getting-started), some algorithms can benefit from improved
  stability or performance due to the preservation of physical properties.
- **Product-formula factors**: A fermionic product formula splits an operator into its groups, so
  the grouping determines both the cost and the Trotter error of a time evolution. See
  [Synthesize an Evolution gate in fermionic space](fermionic_synthesis.html.md#fermionic-synthesis-explanation).

## Usage

The simplest way to define an operator grouping is by setting the target
operator’s `groups` attribute. The example below groups the terms of a small
[`MajoranaOperator`](../stubs/qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator) by hand, collecting the edges of a lattice into lines.

#### HINT
The `qiskit_fermions.operators.terms.grouping` module provides convenience
methods for automatically identifying structure in an operator and assigning
group indices accordingly.

```pycon
>>> edges = [(0, 1), (1, 2), (3, 0), (1, 4), (5, 2), (5, 4), (4, 3)]
>>> groups = [0, 0, 1, 2, 1, 3, 3]
```

([`png`](_downloads/f6414962d9cef933b0e85e11ec75640e/grouping-2.png), [`hires.png`](_downloads/c65520217e69591c3f7768b5754e5266/grouping-2.hires.png), [`pdf`](_downloads/0d33175ce85e6909a97cabd85f9cf5a5/grouping-2.pdf))

![A two-by-three lattice of six Majorana modes, numbered 0 to 5, joined by seven undirected edges. Each edge is labeled with its group: the bottom row (edges 0-1 and 1-2) is group 0, the two outer columns (edges 0-3 and 2-5) are group 1, the middle column (edge 1-4) is group 2, and the top row (edges 3-4 and 4-5) is group 3.](../_images/grouping-2.png)

This is a simple two-by-three lattice of Majorana modes, numbered like the generators of a
[`MajoranaOperator`](../stubs/qiskit_fermions.operators.MajoranaOperator.html.md#qiskit_fermions.operators.MajoranaOperator). Each edge between modes $j$ and $k$ represents the Majorana
bilinear $\gamma_j \gamma_k$, one term of the operator. The goal is to group these terms
according to their edge labels (shown on the graph): the bottom row, the two outer columns, the
middle column, and the top row. Terms connected by edges with the same label are placed in the same
group. The following code shows how that can be done:

### PYTHON

```python
>>> from qiskit_fermions.operators import MajoranaOperator
>>>
>>> # Construct operator directly using sparse arrays
>>> # 7 terms, each with coefficient 1.0, modes from edge endpoints
>>> op = MajoranaOperator(
...     coeffs=[1.0] * 7,
...     modes=[node for edge in edges for node in edge],
...     boundaries=[0, 2, 4, 6, 8, 10, 12, 14],
... )
>>>
>>> # Assign group indices to terms (groups is part of the sparse data structure)
>>> op.groups = groups
>>>
>>> # Partition operator by groups
>>> grouped_ops = op.split_out_groups()
>>>
>>> # Inspect each group
>>> for i, g in enumerate(grouped_ops):
...     print(f"Group {i}: {list(sorted(g.iter_terms()))}")
Group 0: [([0, 1], (1+0j)), ([1, 2], (1+0j))]
Group 1: [([3, 0], (1+0j)), ([5, 2], (1+0j))]
Group 2: [([1, 4], (1+0j))]
Group 3: [([4, 3], (1+0j)), ([5, 4], (1+0j))]
```

### C

```c
#include <qiskit_fermions.h>

uint64_t num_terms = 7;
uint64_t num_modes = 14;
uint32_t modes[14] = {0, 1, 1, 2, 3, 0, 1, 4, 5, 2, 5, 4, 4, 3};
QkComplex64 coeffs[7] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
                         {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[8] = {0, 2, 4, 6, 8, 10, 12, 14};
QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

uint32_t groups[7] = {0, 0, 1, 2, 1, 3, 3};
qf_maj_op_set_groups(op, groups, num_terms);

QfMajoranaOperator *group_ops[4];
qf_maj_op_split_out_groups(op, NULL, 0, group_ops);
```

<a id="grouping-commuting-terms"></a>

## Group commuting terms

The grouping above was written out by hand, because which terms belong together is a choice about
the *problem*. A grouping can also be *derived* from the operator itself. The most useful such
relation is commutation: build the graph whose nodes are the operator’s terms and whose edges join
the pairs that do **not** commute, then color it. A proper coloring places two non-commuting terms
in different groups, so every group ends up mutually commuting.

That graph is the operator’s *clash graph*, and it is the fermionic counterpart of
[`noncommutation_graph()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.PauliList#noncommutation_graph). Commutation relates two whole terms,
so building it is an all-to-all comparison over the terms, independent of how many generators any
one of them multiplies together.

Operators report the clashing pairs directly, and any graph library turns them into a graph:

```pycon
>>> from qiskit_fermions.operators import MajoranaOperator
>>>
>>> op = MajoranaOperator(
...     coeffs=[1.0] * 7,
...     modes=[node for edge in edges for node in edge],
...     boundaries=[0, 2, 4, 6, 8, 10, 12, 14],
... )
>>> clash_edges = op.non_commutation_edges()
>>> clash_edges
[(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 6), (3, 5), (3, 6), (4, 5), (5, 6)]
```

The figure below draws this graph on top of the lattice from the start of this guide, which is shown
in gray. Each term is a white node at the midpoint of its edge, labeled with its bilinear, and a
dashed red line joins every pair of terms that do not commute. Two Majorana bilinears, each on two
distinct modes, anticommute exactly when they share one mode, so every term clashes with the terms
on the edges that meet it at either end. For this operator, the clash graph is therefore the line
graph of the lattice.

([`png`](_downloads/4e07215e807ee48fa8933f3284056a93/grouping-4.png), [`hires.png`](_downloads/bccdd5f6cd8337acb6e991a605328ca5/grouping-4.hires.png), [`pdf`](_downloads/a3617e5c31be1696093648316eadbfa4/grouping-4.pdf))

![The clash graph drawn over the gray lattice from the start of this guide. Each of the seven terms is a white node at the midpoint of its edge, labeled with its bilinear, for example γ₀γ₁. Dashed red lines join the ten pairs of terms that do not commute, which are exactly the pairs of edges that meet at a lattice vertex.](../_images/grouping-4.png)

Coloring it assigns the group indices directly:

### PYTHON

```python
>>> from qiskit_fermions.operators.terms.grouping import (
...     group_terms_by_non_commutation,
... )
>>>
>>> group_terms_by_non_commutation(op)
>>> op.groups
[1, 2, 0, 0, 0, 1, 2]
>>> op.num_groups()
3
```

### C

```c
#include <qiskit_fermions.h>

uint64_t num_terms = 7;
uint64_t num_modes = 14;
uint32_t modes[14] = {0, 1, 1, 2, 3, 0, 1, 4, 5, 2, 5, 4, 4, 3};
QkComplex64 coeffs[7] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
                         {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[8] = {0, 2, 4, 6, 8, 10, 12, 14};
QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

qf_maj_op_group_terms_by_non_commutation(op);
```

The figure below repeats the previous one with this coloring applied. Each term and its edge share
the color of their commuting group, so the lattice shows the grouping the same way the edge labels
show the hand-written groups in the first figure. Every dashed red line joins two different colors:
that is what makes the coloring proper, and it is why every group mutually commutes. Comparing the
two figures also shows where the groupings differ. For example, $\gamma_0 \gamma_1$ and
$\gamma_1 \gamma_2$ share group 0 in the first figure, but they clash and so receive different
colors here. The coloring also needs only three groups where the hand-written grouping uses four.

([`png`](_downloads/4e3ba7eff69d528a5696caaeea6a33e5/grouping-6.png), [`hires.png`](_downloads/9afabbad745e53791a9fe92df5799424/grouping-6.hires.png), [`pdf`](_downloads/bf2db3bc5cd82ba223a421dddde350d2/grouping-6.pdf))

![The clash graph over the lattice, now colored by the three commuting groups, with each edge and its term in the same color. Blue holds γ₃γ₀, γ₁γ₄, and γ₅γ₂; orange holds γ₀γ₁ and γ₅γ₄; green holds γ₁γ₂ and γ₄γ₃. Every dashed red clash line joins two different colors.](../_images/grouping-6.png)

#### IMPORTANT
**The hand-written groups above are not commuting groups.** Their lines resemble the *line flow
sets* of <sup>[1](#id4)</sup>, which do commute, but <sup>[1](#id4)</sup> defines flow sets on transfer operators rather than
on Majorana bilinears. Two bilinears anticommute whenever their edges meet, so group 0 above
contains the anticommuting pair $\gamma_0 \gamma_1$ and $\gamma_1 \gamma_2$. The two
groupings on this page answer different questions, and neither is more correct than the other.

A transfer $T_{jk}$ is *directed*: it acts on the odd generator of its source $j$
and the even generator of its target $k$. Two transfers therefore anticommute only when
they share a source or a target, while a head-to-tail chain shares no generator and commutes.
That is the rule a flow set follows. Orienting each edge as it is listed in `edges` and
writing it as a [`TransferVertexOperator`](../stubs/qiskit_fermions.operators.TransferVertexOperator.html.md#qiskit_fermions.operators.TransferVertexOperator) term turns the hand-written groups into
commuting groups. Listing the clashing pairs inside each hand-written group shows the
difference directly:

```python
>>> from qiskit_fermions.operators import TransferVertexOperator
>>>
>>> def clashes_within_groups(op):
...     return [(a, b) for a, b in op.non_commutation_edges() if groups[a] == groups[b]]
>>>
>>> transfers = TransferVertexOperator(
...     coeffs=[1.0] * 7,
...     left_indices=[j for j, _ in edges],
...     right_indices=[k for _, k in edges],
...     boundaries=list(range(8)),
... )
>>> clashes_within_groups(op)  # the Majorana bilinears
[(0, 1), (5, 6)]
>>> clashes_within_groups(transfers)
[]
```

For the bilinears, the two pairs are the head-to-tail chains of group 0
($\gamma_0 \gamma_1$ and $\gamma_1 \gamma_2$) and of group 3
($\gamma_5 \gamma_4$ and $\gamma_4 \gamma_3$). Written as transfers, those same
chains commute, and no group contains a clash.

Only for transfers does the direction of an edge matter. For Majorana bilinears, reversing an
edge only flips the sign of its term, which is why this guide draws the lattice without arrows.

#### NOTE
Indices refer to the operator’s *current* term order, which is not preserved across every
transformation. Build the operator from its sparse arrays, as above, or apply
[`canonical_order()`](../stubs/qiskit_fermions.operators.terms.canonical_order.html.md#qiskit_fermions.operators.terms.canonical_order) first, when the result has to be reproducible.

Every term counts as written, except that a term with an exactly zero coefficient clashes with
nothing. Simplify an operator before computing its clash graph: that removes near-zero and
duplicate terms, which would otherwise inflate the graph. Simplifying can reorder the terms, so
compute the indices on the simplified operator.

The coloring is a greedy heuristic, so the number of groups is an upper bound on the smallest
achievable rather than a minimum.

## Analyze a grouping

Groups prescribe no meaning of their own: the groups array records which terms belong together,
never why. No routine in this library reports whether a grouping is “correct”, because there is no
single notion of correctness to report against. Downstream consumers do, however, assume properties
of a grouping, and the functions in this section let you check such an assumption up front rather
than paying for it on every call.

The most common convention is that each group is separately Hermitian. That is what makes fermionic
randomized product formulas possible at all: a lone $a^\dagger_p a_q$ with $p \neq q$
is not Hermitian, so its time evolution is not unitary and it cannot be sampled on its own. Pairing
it with $a^\dagger_q a_p$ yields a group that can be.

### PYTHON

```python
>>> from qiskit_fermions.operators import FermionOperator
>>> from qiskit_fermions.operators.terms.grouping import (
...     group_coeff_means,
...     groups_are_hermitian,
...     groups_have_uniform_coeffs,
... )
>>>
>>> # a conjugate pair, plus one unpaired term. The operator is built from the sparse
>>> # arrays rather than from a dictionary, because only the former fixes the term order
>>> # that the group indices below are paired with.
>>> pair_and_single = FermionOperator(
...     [1.0, 1.0, 1.0],
...     [True, False, True, False, True, False],
...     [0, 1, 1, 0, 2, 3],
...     [0, 2, 4, 6],
... )
>>> pair_and_single.groups = [0, 0, 1]
>>>
>>> groups_are_hermitian(pair_and_single)  # group 1 is unpaired
[True, False]
>>> group_coeff_means(pair_and_single)
[1.0, 1.0]
```

### C

```c
#include <qiskit_fermions.h>

uint64_t num_terms = 3;
uint64_t num_actions = 6;
bool actions[6] = {true, false, true, false, true, false};
uint32_t modes[6] = {0, 1, 1, 0, 2, 3};
QkComplex64 coeffs[3] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
uint32_t boundaries[4] = {0, 2, 4, 6};
QfFermionOperator *op =
    qf_ferm_op_new(num_terms, num_actions, coeffs, actions, modes, boundaries);

uint32_t groups[3] = {0, 0, 1};
qf_ferm_op_set_groups(op, groups, num_terms);

bool hermitian[2];
qf_ferm_op_groups_are_hermitian(op, 1e-8, hermitian);

double means[2];
qf_ferm_op_group_coeff_means(op, means);
```

### One group index per term is enforced

The groups array holds one index per term, and assigning one of a different length raises. Nothing
downstream re-checks the invariant, so it is enforced where the indices enter the operator: an array
that is too short would silently drop trailing terms, and one that is too long would invent groups
that no term carries.

```python
>>> pair_and_single.groups = [0, 0]  # one index short, on a three-term operator
Traceback (most recent call last):
ValueError: expected one group index per term, but got 2 for 3 terms
```

Clearing the group indices by assigning `None` is always allowed.

### The two checks are independent

[`groups_have_uniform_coeffs()`](../stubs/qiskit_fermions.operators.terms.groups_have_uniform_coeffs.html.md#qiskit_fermions.operators.terms.groups_have_uniform_coeffs) reports whether a
group’s coefficients agree, which is the assumption
[`group_coeff_means()`](../stubs/qiskit_fermions.operators.terms.group_coeff_means.html.md#qiskit_fermions.operators.terms.group_coeff_means) makes when it averages
magnitudes. It is not a weaker form of the Hermiticity check: neither check implies the other, in
either direction.

```pycon
>>> from qiskit_fermions.operators.terms.grouping import (
...     groups_are_hermitian,
...     groups_have_uniform_coeffs,
... )
>>> # uniform coefficients, but not Hermitian: the terms are not each other's adjoint
>>> uniform = FermionOperator(
...     [1.0, 1.0],
...     [True, False, True, False],
...     [0, 1, 2, 3],
...     [0, 2, 4],
... )
>>> uniform.groups = [0, 0]
>>> groups_have_uniform_coeffs(uniform), groups_are_hermitian(uniform)
([True], [False])
```

```pycon
>>> # Hermitian, but with mixed magnitudes: two conjugate pairs at different scales
>>> mixed = FermionOperator(
...     [1.0, 1.0, 5.0, 5.0],
...     [True, False, True, False, True, False, True, False],
...     [0, 1, 1, 0, 2, 3, 3, 2],
...     [0, 2, 4, 6, 8],
... )
>>> mixed.groups = [0, 0, 0, 0]
>>> groups_are_hermitian(mixed), groups_have_uniform_coeffs(mixed)
([True], [False])
```

By default the magnitudes are compared, because that is what
[`group_coeff_means()`](../stubs/qiskit_fermions.operators.terms.group_coeff_means.html.md#qiskit_fermions.operators.terms.group_coeff_means) averages. A Hermitian conjugate
pair with complex coefficients has equal magnitudes but unequal coefficients, so it satisfies only
the default form.

```pycon
>>> conjugate = FermionOperator(
...     [1.0j, -1.0j],
...     [True, False, True, False],
...     [0, 1, 1, 0],
...     [0, 2, 4],
... )
>>> conjugate.groups = [0, 0]
>>> groups_have_uniform_coeffs(conjugate)
[True]
>>> groups_have_uniform_coeffs(conjugate, abs=False)
[False]
```

#### NOTE
[`groups_are_hermitian()`](../stubs/qiskit_fermions.operators.terms.groups_are_hermitian.html.md#qiskit_fermions.operators.terms.groups_are_hermitian) inherits the one-sided
guarantee of [`is_hermitian()`](../stubs/qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.is_hermitian): a `True` entry is
always reliable, while a `False` entry is reliable only for operator types whose normal form is
a genuine canonical form. Consult the specific operator type to find out which applies.

* <a id='id4'>**[1]**</a> A. Gandon et al., Stabilizer-based quantum simulation of fermion dynamics with local qubit encodings, [arXiv:2512.11418v2](https://arxiv.org/abs/2512.11418v2).
