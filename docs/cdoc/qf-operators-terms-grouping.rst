==================
Grouping Functions
==================

.. meta::
   :description: C API functions that assign operator terms to groups automatically and that analyze
                 an existing grouping.

Refer to :ref:`grouping_explanation` for a detailed explanation of this module's
functionality.

Assignment
^^^^^^^^^^

Rather than always relying on the user to provide the group indices themselves,
the C API provides a collection of functions which determine the grouping
information automatically.

Unlike the analysis routines below, these are provided per operator
representation only where the grouping makes sense for it: the
electronic-structure grouping is specific to :c:struct:`QfFermionOperator`, while
the commuting grouping exists for every representation.

.. table::

  ========================================================= ==============================================================
  :c:func:`qf_ferm_op_group_terms_by_electronic_structure`  Groups the terms of an operator by their electronic structure.
  :c:func:`qf_op_type_group_terms_by_non_commutation`       Groups the terms of an operator into mutually commuting sets.
  ========================================================= ==============================================================

Commuting groups
................

.. c:function:: QfExitCode qf_op_type_group_terms_by_non_commutation(OpType *op)

    .. caution::
       The function signature here is **generic**! A real one will replace
       ``OpType`` with an actual :ref:`operator representation <qf_operators>`
       and ``op_type`` with its matching prefix (for example, ``ferm_op`` for
       :c:struct:`QfFermionOperator`).

    Populates the ``groups`` attribute such that any two terms which do **not**
    commute receive different group indices, replacing any grouping the operator
    already carried.

    It does so by coloring the operator's *clash graph* - one node per term, one
    edge per non-commuting pair. That graph is available in its own right through
    the per-representation accessors, documented alongside their operators:
    :c:func:`qf_ferm_op_non_commutation_edges`,
    :c:func:`qf_maj_op_non_commutation_edges`,
    :c:func:`qf_edge_op_non_commutation_edges` and
    :c:func:`qf_transfer_op_non_commutation_edges`.

    A term with an exactly zero coefficient clashes with nothing, while every
    other term counts as written. Simplify the operator first (with
    ``qf_op_type_simplify``) to remove near-zero and duplicate terms, which would
    otherwise inflate the number of groups.

    .. note::
       The coloring is the greedy *largest-first* heuristic (terms with the most
       clashes are placed first), so the resulting number of groups is an upper
       bound on the smallest achievable rather than a minimum. The graph itself is
       never built: memory grows linearly with the number of terms, where the edge
       array of ``qf_op_type_non_commutation_edges`` grows quadratically.

    .. important::
       Whether these commuting groups agree with the line flow sets used to
       compile the time evolution of lattice Hamiltonians depends on the operator
       representation, not on the lattice. A flow set forbids two arrows that
       share a source or a target while permitting a head-to-tail chain. A
       transfer :math:`T_{jk}` acts on the odd generator of its source and the
       even generator of its target, so for the hopping terms of a
       :c:struct:`QfTransferVertexOperator` the clash relation *is* that rule.
       Majorana and edge-vertex bilinears instead act on the same generator of a
       shared mode however they meet, so there a head-to-tail chain does **not**
       commute and the two partitions diverge.

    .. note::
       For simplicity, the example below shows only the
       :c:struct:`QfMajoranaOperator` variant. The other operator representations
       follow the same pattern.

    .. code-block:: c
       :linenos:

       uint64_t num_terms = 3;
       uint64_t num_modes = 6;
       uint32_t modes[6] = {0, 1, 1, 2, 3, 4};
       QkComplex64 coeffs[3] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
       uint32_t boundaries[4] = {0, 2, 4, 6};
       QfMajoranaOperator *op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

       // Terms 0 and 1 share Majorana mode 1, so they land in different groups.
       QfExitCode exit = qf_maj_op_group_terms_by_non_commutation(op);
       assert(exit == QfExitCode_Success);
       assert(qf_maj_op_num_groups(op) == 2);

       qf_maj_op_free(op);

    :param op: A pointer to the operator whose terms to group.
    :return: An exit code. ``QfExitCode_ValueError`` is returned exactly when
        ``qf_op_type_non_commutation_edges`` returns it: if ``op`` has more than
        :math:`2^{32}` terms, or is a :c:struct:`QfFermionOperator` with a term of
        degree above 20 and a non-zero coefficient. The operator's groups are then
        left untouched.

Analysis
^^^^^^^^

Group indices carry no intrinsic meaning: they only say which terms belong
together, never why. Accordingly, none of the functions below reports whether a
grouping is "correct" and there is no single notion of correctness to report
against. Each answers one narrow, stated question about an existing grouping,
leaving the interpretation to you. They are provided so that an assumption a
downstream consumer makes about a grouping can be checked up front, rather than
being paid for on every call.

Since these functions work with generic operator representations (a notion that
does not exist in C), they are explained generically in the following three
sections. The actual functions contained in the C API for the various operator
representations are listed at the bottom of this page.

Every one of them requires the operator to track group indices. Check this via
``qf_op_type_has_groups`` first; calling them on an operator without groups
aborts.

Group coefficient means
.......................

.. c:function:: void qf_op_type_group_coeff_means(const OpType *op, double *means_out)

    .. caution::
       The function signature here is **generic**! A real one will replace
       ``OpType`` with an actual :ref:`operator representation <qf_operators>`
       and ``op_type`` with its matching prefix (for example, ``ferm_op`` for
       :c:struct:`QfFermionOperator`).

    The ``i``-th entry is the sum of ``abs(coeff)`` over the terms in group
    ``i``, divided by the number of terms in that group. This is the sampling
    weight of a randomized product formula (for example, qDRIFT) that draws
    whole groups rather than individual terms: it is the magnitude of one
    *atomic* group, which is the relevant scale because grouping is what makes
    each sampled piece Hermitian, and hence its time evolution unitary, in the
    first place.

    It is computed in a single pass over the operator rather than by reducing
    ``qf_op_type_get_coeffs`` and ``qf_op_type_get_groups`` (one value per
    *ungrouped* term each) on the caller's side.

    .. note::
       A group index that no term carries weighs ``0.0``, which keeps it out of
       the sample.

    :param op: A pointer to the operator whose groups to reduce.
    :param means_out: A pointer to the array of doubles into which to write the
        means. Must be sized to ``qf_op_type_num_groups``.

Group hermiticity
.................

.. c:function:: void qf_op_type_groups_are_hermitian(const OpType *op, double atol, bool *hermitian_out)

    .. caution::
       The function signature here is **generic**! See the preceding caution.

    Groups prescribe no meaning of their own; this checks one *common*
    convention, namely that each group is separately Hermitian. That is the
    property a randomized product formula relies on when it samples whole
    groups, since only a Hermitian group has a unitary time evolution. An empty
    group counts as Hermitian, because the zero operator is.

    .. note::
       This inherits the one-sided guarantee of ``qf_op_type_is_hermitian``: a
       ``true`` entry is always reliable, while a ``false`` entry is reliable
       only for operator types whose normal form is a genuine canonical form.
       For the others the check is *conservative* and can report ``false`` for a
       group that is in fact Hermitian. Consult the specific operator
       representation to find out which applies.

    .. note::
       This neither implies nor is implied by
       ``qf_op_type_groups_have_uniform_coeffs``. Uniform coefficients do not
       make a group Hermitian, and a Hermitian group can mix magnitudes.

    :param op: A pointer to the operator whose groups to check.
    :param atol: The absolute tolerance upto which coefficients are considered
        equal.
    :param hermitian_out: A pointer to the array of booleans into which to write
        one flag per group. Must be sized to ``qf_op_type_num_groups``.

Group coefficient uniformity
............................

.. c:function:: void qf_op_type_groups_have_uniform_coeffs(const OpType *op, double atol, bool abs, bool *uniform_out)

    .. caution::
       The function signature here is **generic**! See the preceding caution.

    With ``abs``, the coefficient *magnitudes* are compared, which is the
    assumption that ``qf_op_type_group_coeff_means`` makes when it averages
    ``abs(coeff)``. Without it, the coefficients must match exactly. Note that a
    Hermitian group can legitimately fail the stricter form, since a conjugate
    pair with complex coefficients has equal magnitudes but unequal
    coefficients. An empty or single-term group is trivially uniform.

    .. note::
       This is *not* a weaker form of ``qf_op_type_groups_are_hermitian``:
       neither implies the other. Uniform coefficients do not make a group
       Hermitian, and a Hermitian group can mix magnitudes.

    :param op: A pointer to the operator whose groups to check.
    :param atol: The absolute tolerance upto which coefficients are considered
        equal.
    :param abs: Whether to compare coefficient magnitudes rather than the
        coefficients themselves.
    :param uniform_out: A pointer to the array of booleans into which to write
        one flag per group. Must be sized to ``qf_op_type_num_groups``.

Example
.......

.. code-block:: c
   :linenos:

   QfFermionOperator *op = qf_ferm_op_zero();
   bool actions[2] = {true, false};
   QkComplex64 coeff = {1.0, 0.0};
   uint32_t modes_a[2] = {0, 1};
   qf_ferm_op_add_term(op, 2, actions, modes_a, &coeff);
   uint32_t modes_b[2] = {1, 0};
   qf_ferm_op_add_term(op, 2, actions, modes_b, &coeff);

   uint32_t groups_in[2] = {0, 0};
   assert(qf_ferm_op_set_groups(op, groups_in, 2) == QfExitCode_Success);

   double means[1];
   qf_ferm_op_group_coeff_means(op, means);
   assert(means[0] == 1.0);

   bool hermitian[1];
   qf_ferm_op_groups_are_hermitian(op, 1e-8, hermitian);
   assert(hermitian[0]);

   bool uniform[1];
   qf_ferm_op_groups_have_uniform_coeffs(op, 1e-8, true, uniform);
   assert(uniform[0]);

Functions
^^^^^^^^^

.. doxygengroup:: qf_operator_terms_grouping
   :content-only:
   :members:
   :undoc-members:
