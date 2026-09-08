.. _grouping_explanation:

Group operator terms: use the operator structure
================================================

Grouping is one of the operations under :mod:`qiskit_fermions.operators.terms`,
which collects routines that partition the terms of an operator based on their
structure.

As described in the :ref:`operators guide <term_grouping>`, operators can store a
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
  structures relevant to quantum simulation, as shown in [1]_.

- **Improved algorithmic stability**: As demonstrated in the :ref:`SqDRIFT routine
  <sqdrift_getting_started>`, some algorithms can benefit from improved
  stability or performance due to the preservation of physical properties.


Usage
-----

The simplest way to define an operator grouping is by setting the target
operator's ``groups`` attribute. Below is an example for grouping terms
according to the *line flow sets* as shown in Figure 1c of [1]_.

.. hint::
   The :mod:`qiskit_fermions.operators.terms.grouping` module provides convenience
   methods for automatically identifying structure in an operator and assigning
   group indices accordingly.

.. plot::
   :alt: A simple directed graph on which to define a MajoranaOperator.
   :context: close-figs
   :include-source:

    >>> import rustworkx as rx
    >>> graph = rx.PyDiGraph()
    >>> _ = graph.add_nodes_from(range(6))
    >>> edges = [(0, 1), (1, 2), (3, 0), (1, 4), (5, 2), (5, 4), (4, 3)]
    >>> groups = [0, 0, 1, 2, 1, 3, 3]
    >>> edges_with_group_as_payload = [(i, j, g) for (i, j), g in zip(edges, groups)]
    >>> _ = graph.add_edges_from(edges_with_group_as_payload)
    >>> from rustworkx.visualization import mpl_draw
    >>> mpl_draw(
    ...     graph,
    ...     pos={i: (i % 3, i // 3) for i in range(6)},
    ...     edge_labels=str,
    ...     with_labels=True,
    ... )
    <Figure size ... with 1 Axes>

This is a simple two-by-three lattice of Majorana modes connected by
directed edges. Each edge represents a term in the operator acting on the two
connected modes. The goal is to group these terms according to their edge labels
(shown on the graph). Terms connected by edges with the same label are placed in
the same group. The following code shows how that can be done:

.. tab-set-code::

    .. code-block:: python

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

    .. code-block:: c

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


Analyze a grouping
------------------

Groups prescribe no meaning of their own: the groups array records which terms belong together,
never why. No routine in this library reports whether a grouping is "correct", because there is no
single notion of correctness to report against. Downstream consumers do, however, assume properties
of a grouping, and the functions in this section let you check such an assumption up front rather
than paying for it on every call.

The most common convention is that each group is separately Hermitian. That is what makes fermionic
randomized product formulas possible at all: a lone :math:`a^\dagger_p a_q` with :math:`p \neq q`
is not Hermitian, so its time evolution is not unitary and it cannot be sampled on its own. Pairing
it with :math:`a^\dagger_q a_p` yields a group that can be.

.. tab-set-code::

    .. code-block:: python

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

    .. code-block:: c

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


One group index per term is enforced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The groups array holds one index per term, and assigning one of a different length raises. Nothing
downstream re-checks the invariant, so it is enforced where the indices enter the operator: an array
that is too short would silently drop trailing terms, and one that is too long would invent groups
that no term carries.

.. code-block:: python

   >>> pair_and_single.groups = [0, 0]  # one index short, on a three-term operator
   Traceback (most recent call last):
   ValueError: expected one group index per term, but got 2 for 3 terms

Clearing the group indices by assigning ``None`` is always allowed.

The two checks are independent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~qiskit_fermions.operators.terms.grouping.groups_have_uniform_coeffs` reports whether a
group's coefficients agree, which is the assumption
:func:`~qiskit_fermions.operators.terms.grouping.group_coeff_means` makes when it averages
magnitudes. It is not a weaker form of the Hermiticity check: neither check implies the other, in
either direction.

.. plot::
   :context:
   :nofigs:
   :include-source:

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

By default the magnitudes are compared, because that is what
:func:`~qiskit_fermions.operators.terms.grouping.group_coeff_means` averages. A Hermitian conjugate
pair with complex coefficients has equal magnitudes but unequal coefficients, so it satisfies only
the default form.

.. plot::
   :context:
   :nofigs:
   :include-source:

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

.. note::
   :func:`~qiskit_fermions.operators.terms.grouping.groups_are_hermitian` inherits the one-sided
   guarantee of :meth:`~qiskit_fermions.operators.OperatorTrait.is_hermitian`: a ``True`` entry is
   always reliable, while a ``False`` entry is reliable only for operator types whose normal form is
   a genuine canonical form. Consult the specific operator type to find out which applies.


.. [1] A. Gandon et al., Stabilizer-based quantum simulation of fermion dynamics
       with local qubit encodings, `arXiv:2512.11418v2
       <https://arxiv.org/abs/2512.11418v2>`_.
