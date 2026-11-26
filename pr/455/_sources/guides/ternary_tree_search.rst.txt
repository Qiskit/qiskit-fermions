.. _ternary_tree_search_explanation:

Choose a ternary tree
=====================

The :ref:`mappers guide <mappers_explanation>` introduces ternary trees and shows that Jordan-Wigner
and the balanced tree are two shapes out of the same family. That leaves a practical question open:
which tree should you use?

The balanced tree keeps the heaviest single Majorana operator as light as possible. A
simulation, however, does not pay for single operators. It pays for the mapped Hamiltonian, whose
terms are products of several Majorana operators, and how heavy those products become depends on how
the tree arranges the operators that appear together. A tree chosen without looking at the
Hamiltonian cannot account for that.

This guide measures that cost on a small molecule, grows a tree adapted to the Hamiltonian with the
Hamiltonian-adaptive ternary tree (HATT) algorithm [1]_, and then improves on it with a local search
over trees.


Measure the cost of a tree
--------------------------

Load the electronic-structure Hamiltonian of a water molecule in the STO-3G basis. Its seven spatial
orbitals give 14 fermionic modes.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> from qiskit_fermions.operators import FermionOperator
         >>> from qiskit_fermions.operators.library import FCIDump
         >>>
         >>> fcidump = FCIDump.from_file("docs/guides/h2o.fcidump")
         >>> hamiltonian = FermionOperator.from_fcidump(fcidump)
         >>> num_modes = 2 * fcidump.norb
         >>> num_modes
         14

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         #include <assert.h>
         #include <inttypes.h>
         #include <stdio.h>
         #include <qiskit_fermions.h>

         QfFCIDump *fcidump;
         QfExitCode exit = qf_fcidump_from_file("docs/guides/h2o.fcidump", &fcidump);
         assert(exit == QfExitCode_Success);
         QfFermionOperator *hamiltonian = qf_ferm_op_from_fcidump(fcidump);
         uint32_t num_modes = 2 * qf_fcidump_norb(fcidump);  // 14
         qf_fcidump_free(fcidump);

The metric used throughout is the **total Pauli weight**: the number of qubits each term of the
mapped Hamiltonian acts on, summed over all terms. It is a proxy for the entangling gates and basis
changes that one Trotter step costs.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
         >>> from qiskit_fermions.mappers.library import fermion_ternary_tree
         >>>
         >>> def mapped(operator, tree):
         ...     """Maps ``operator`` through ``tree`` and drops numerically vanishing terms."""
         ...     encoding = TernaryTreeEncoding(tree)
         ...     return fermion_ternary_tree(operator, encoding, tree.num_nodes).simplify(1e-10)
         >>>
         >>> def total_weight(observable):
         ...     """The summed Pauli weight of the terms of ``observable``."""
         ...     return sum(len(observable[i].indices) for i in range(len(observable)))
         >>>
         >>> def report(name, tree):
         ...     print(
         ...         f"{name:14s} max weight {tree.max_weight:2d}"
         ...         f" | total weight {total_weight(mapped(hamiltonian, tree)):5d}"
         ...     )

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // Map the Hamiltonian through a given `QfTernaryTree *tree`.
         QfTernaryTreeEncoding *encoding;
         exit = qf_ternary_tree_encoding_new(tree, NULL, &encoding);
         assert(exit == QfExitCode_Success);
         QkObs *raw;
         uint32_t num_qubits = qf_ternary_tree_num_nodes(tree);
         exit = qf_ferm_op_ternary_tree(hamiltonian, encoding, num_qubits, &raw);
         assert(exit == QfExitCode_Success);

         // The C counterpart of `simplify`: sum equal terms and drop the vanishing ones.
         QkObs *observable = qk_obs_canonicalize(raw, 1e-10);

         qk_obs_free(observable);
         qk_obs_free(raw);
         qf_ternary_tree_encoding_free(encoding);

The call to ``simplify`` matters, for two reasons. The integrals in this file include entries of
order ``1e-15`` that vanish by symmetry, and the terms they produce would otherwise count as much
as any other. Terms that cancel in exact arithmetic can also survive the mapping as floating-point
residue of order ``1e-16``, and whether a given one survives depends on the order of summation, so
without dropping them the term count and the total weight vary slightly from run to run.

Compare the two trees from the mappers guide:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> jordan_wigner = TernaryTree.chain(num_modes, "Z")
         >>> balanced = TernaryTree.breadth_first(num_modes, 3)
         >>> report("Jordan-Wigner", jordan_wigner)
         Jordan-Wigner  max weight 14 | total weight  6332
         >>> report("balanced", balanced)
         balanced       max weight  4 | total weight  6762

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         QfTernaryTree *jordan_wigner, *balanced;
         exit = qf_ternary_tree_chain(num_modes, QfPauliLabel_Z, &jordan_wigner);
         assert(exit == QfExitCode_Success);
         QfPauliLabel order[3] = {QfPauliLabel_X, QfPauliLabel_Y, QfPauliLabel_Z};
         exit = qf_ternary_tree_breadth_first(num_modes, 3, order, &balanced);
         assert(exit == QfExitCode_Success);
         printf("%" PRIu32 "\n", qf_ternary_tree_max_weight(jordan_wigner));  // 14
         printf("%" PRIu32 "\n", qf_ternary_tree_max_weight(balanced));       // 4

.. plot::
   :context:
   :nofigs:

   >>> jordan_wigner_total = total_weight(mapped(hamiltonian, jordan_wigner))
   >>> balanced_total = total_weight(mapped(hamiltonian, balanced))
   >>> bool(balanced.max_weight < jordan_wigner.max_weight)
   True
   >>> bool(jordan_wigner_total < balanced_total)
   True

The balanced tree has by far the lower worst single-operator weight, yet it maps the Hamiltonian
onto a heavier observable than Jordan-Wigner does. The two Majorana operators of a mode meet on that
mode's qubit. In the Jordan-Wigner chain they end there, so a number operator maps to a single
``Z``. In the balanced tree, at every node with children at least one of them continues down into a
subtree, so the number operator of such a mode acts on several qubits, and many terms of a
molecular Hamiltonian are number operators or their products. Judged by ``max_weight`` alone, the
balanced tree looks like the better choice; judged by the Hamiltonian you actually simulate, it is
not.


Adapt the tree to the Hamiltonian
---------------------------------

HATT builds the tree around the Hamiltonian instead. It works bottom-up: every leg starts out as a
subtree of its own, and each step creates one qubit, choosing two of its children (the third is
then fixed, so that the two Majorana operators of each mode stay paired) such that the
Hamiltonian's Pauli weight on that qubit is as small as possible. Summed over all steps, these costs
are the total Pauli weight of the Majorana terms you pass in, each term counted on its own. If the
Majorana form is normal-ordered and simplified with the same tolerance first, its distinct terms map
to distinct Pauli strings, and that sum is exactly the metric used here, which is also how Ref. [1]_
scores it. This guide converts the Hamiltonian that way:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> from qiskit_fermions.mappers.library import majorana_operator
         >>> from qiskit_fermions.mappers.optimization import hatt_tree
         >>>
         >>> majorana = majorana_operator(hamiltonian).normal_ordered().simplify(1e-10)
         >>> hatt = hatt_tree(majorana, num_modes)
         >>> report("HATT", hatt)
         HATT           max weight  7 | total weight  5545

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // Convert, normal-order and simplify, as in the Python step.
         QfMajoranaOperator *converted = qf_fermion_to_majorana(hamiltonian);
         QfMajoranaOperator *ordered = qf_maj_op_normal_ordered(converted, false, true);
         QfMajoranaOperator *majorana = qf_maj_op_simplify(ordered, 1e-10);
         qf_maj_op_free(ordered);
         qf_maj_op_free(converted);
         QfTernaryTree *hatt;
         exit = qf_ternary_tree_hatt(majorana, num_modes, false, &hatt);
         assert(exit == QfExitCode_Success);

.. plot::
   :context:
   :nofigs:

   >>> hatt_total = total_weight(mapped(hamiltonian, hatt))
   >>> bool(hatt_total < jordan_wigner_total < balanced_total)
   True
   >>> bool(balanced.max_weight < hatt.max_weight < jordan_wigner.max_weight)
   True
   >>> bool(total_weight(mapped(hamiltonian, hatt_tree(hamiltonian, num_modes))) < hatt_total)
   True

HATT gives the lowest total weight of the three, although its worst-case weight is higher than the
balanced tree's. The function also accepts the fermionic Hamiltonian, or any other operator
implementing :class:`.SupportsMajoranaOperator`, and converts it to its Majorana form for you. That
conversion keeps every term as it comes, so HATT then also counts terms that would cancel or vanish,
and optimizes a slightly different objective. Because the construction is greedy, that can land on
a tree that is a little lighter or a little heavier by the metric here; on this Hamiltonian it is
lighter. The mode count is inferred from the operator's support if you omit it.

.. note::
   The Jordan-Wigner and HATT totals above reproduce the values that Ref. [1]_ reports for this
   molecule, which suggests the Hamiltonian has the same term structure as the paper's. The
   balanced tree here is not the paper's balanced construction, so its total differs.

.. warning::
   HATT is a greedy heuristic, so it is not guaranteed to find the lowest-weight tree, and Pauli
   weight is not the circuit cost. Ref. [1]_ reports cases where Jordan-Wigner gives fewer CNOT
   gates or a shallower circuit. Measure on your own Hamiltonian.

Choose the cost
~~~~~~~~~~~~~~~

The published cost counts terms and ignores their coefficients. The coefficients of this Hamiltonian
span several orders of magnitude, so a term with a tiny coefficient counts as much as a dominant
one. Passing ``weighted=True`` weights each term by the magnitude of its coefficient instead. To see
the difference, weight the metric the same way:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> def weighted_weight(observable):
         ...     """Like ``total_weight``, but each term weighted by ``|coefficient|``."""
         ...     return sum(
         ...         abs(observable[i].coeff) * len(observable[i].indices)
         ...         for i in range(len(observable))
         ...     )
         >>>
         >>> weighted = hatt_tree(majorana, num_modes, weighted=True)
         >>> candidates = {"Jordan-Wigner": jordan_wigner, "HATT": hatt, "HATT weighted": weighted}
         >>> for name, tree in candidates.items():
         ...     print(f"{name:14s} {weighted_weight(mapped(hamiltonian, tree)):6.1f}")
         Jordan-Wigner   158.7
         HATT            164.3
         HATT weighted   137.4
         >>> report("HATT weighted", weighted)
         HATT weighted  max weight 14 | total weight  5620

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         QfTernaryTree *weighted;
         exit = qf_ternary_tree_hatt(majorana, num_modes, true, &weighted);
         assert(exit == QfExitCode_Success);

.. plot::
   :context:
   :nofigs:

   >>> by_coefficient = {
   ...     name: weighted_weight(mapped(hamiltonian, tree)) for name, tree in candidates.items()
   ... }
   >>> others = min(by_coefficient["Jordan-Wigner"], by_coefficient["HATT"])
   >>> bool(by_coefficient["HATT weighted"] < others)
   True
   >>> bool(hatt_total < total_weight(mapped(hamiltonian, weighted)))
   True
   >>> bool(by_coefficient["Jordan-Wigner"] < by_coefficient["HATT"])
   True

Measured by coefficient, the published cost does worse than Jordan-Wigner on this Hamiltonian,
while the weighted variant does clearly better. Each variant wins on the metric it optimizes, so
choose the one that matches what you pay for. The weighted cost is a deviation from the published
algorithm, though, so the default matches the paper.


Search over trees
-----------------

HATT constructs a tree in one pass and never revisits a choice. The alternative is to search: start
somewhere and make small changes that keep the tree a valid encoding, keeping those that help.

:class:`.TernaryTree` provides these edits. Each returns a new tree, and an invalid move raises an
error rather than producing an invalid tree:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Edit
     - Effect
   * - :meth:`~.TernaryTree.reparent`
     - Move a node, and the subtree below it, onto another node's free link.
   * - :meth:`~.TernaryTree.reroot`
     - Make a given node the root, reversing the links along the path to it.
   * - :meth:`~.TernaryTree.relabel`
     - Permute the ``X``, ``Y``, and ``Z`` labels on one node's three links.

:meth:`~.TernaryTree.leg_labels` reports which of a node's links end in a leg and can therefore
accept a subtree, so you can propose a move without guessing:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> # Every node of a chain but the last spends only its Z link on the next node.
         >>> TernaryTree.chain(4, "Z").leg_labels(0)
         ['X', 'Y']
         >>> # The root of the HATT tree has a child on each of its three links.
         >>> hatt.leg_labels(hatt.root)
         []

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // Every node of a chain but the last spends only its Z link on the next node.
         QfTernaryTree *chain;
         exit = qf_ternary_tree_chain(4, QfPauliLabel_Z, &chain);
         assert(exit == QfExitCode_Success);
         QfPauliLabel free_labels[3];
         uint64_t num_free;
         exit = qf_ternary_tree_leg_labels(chain, 0, free_labels, &num_free);  // X and Y
         assert(exit == QfExitCode_Success);
         qf_ternary_tree_free(chain);
         // The C API does not expose the root of a tree, so the HATT check has no counterpart.

That is enough for a hill climb. Start from the HATT tree, propose random reparentings, and keep
every one that lowers the total weight:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> import random
         >>>
         >>> rng = random.Random(42)
         >>> start = total_weight(mapped(hamiltonian, hatt))
         >>> tree, best = hatt, start
         >>> for _ in range(200):
         ...     node, parent = rng.randrange(num_modes), rng.randrange(num_modes)
         ...     available = tree.leg_labels(parent)
         ...     if node in (tree.root, parent) or not available:
         ...         continue  # the root cannot be reparented, nor a node attached to itself
         ...     try:
         ...         candidate = tree.reparent(node, parent, rng.choice(available))
         ...     except ValueError:
         ...         continue  # the move would have created a cycle
         ...     cost = total_weight(mapped(hamiltonian, candidate))
         ...     if cost < best:
         ...         tree, best = candidate, cost
         >>> bool(best < start)
         True

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // One proposed move: hang `node`, with its subtree, off a free link of `parent`.
         QfPauliLabel available[3];
         uint64_t num_available;
         exit = qf_ternary_tree_leg_labels(hatt, parent, available, &num_available);
         assert(exit == QfExitCode_Success);
         if (num_available > 0) {
             QfTernaryTree *candidate;
             exit = qf_ternary_tree_reparent(hatt, node, parent, available[0], &candidate);
             if (exit == QfExitCode_Success) {
                 // Map the Hamiltonian through `candidate` as shown earlier and keep it if its
                 // total weight is lower.
                 qf_ternary_tree_free(candidate);
             }
             // Otherwise the move is invalid, for example because `node` is the root or the move
             // would create a cycle, and the result is QfExitCode_ValueError.
         }

         // Free what the earlier C tabs created.
         qf_ternary_tree_free(weighted);
         qf_ternary_tree_free(hatt);
         qf_ternary_tree_free(balanced);
         qf_ternary_tree_free(jordan_wigner);
         qf_maj_op_free(majorana);
         qf_ferm_op_free(hamiltonian);

A few accepted moves lower the total weight below HATT's, which shows that the greedy construction
does not find every improvement and that the edits can find some of the rest. Each candidate costs a
full mapping of the Hamiltonian, though, so a search is far more expensive than a one-pass
construction. Starting from a good tree, as here, is what keeps it short.

The objective is yours to choose. Swapping ``total_weight`` for ``weighted_weight`` is a one-line
change, and so is an objective that no construction on this page considers, such as how many mapped
terms act on qubits that are connected on your device.

Treespilation [2]_ is a proposed method that formalizes a search of this kind over ternary trees,
with the CNOT count of a state-preparation circuit on a device as its objective.


Next steps
----------

- The :ref:`mappers guide <mappers_explanation>` covers what a ternary tree is, how the operator
  representations map through one, and the total-parity operator.
- :mod:`qiskit_fermions.mappers.optimization` also holds excitation-span minimization for circuit
  synthesis.
- The C API provides the same functionality: :c:func:`qf_ternary_tree_hatt` grows a tree, and
  :c:func:`qf_ternary_tree_reparent`, :c:func:`qf_ternary_tree_reroot`,
  :c:func:`qf_ternary_tree_relabel`, and :c:func:`qf_ternary_tree_leg_labels` edit one.

.. [1] Y. Liu, K. Yao, J. Hong, J. Froustey, E. Rrapaj, C. Iancu, G. Li, and Y. Shi, HATT:
       Hamiltonian Adaptive Ternary Tree for Optimizing Fermion-to-Qubit Mapping (2024),
       `arXiv:2409.02010 <https://arxiv.org/abs/2409.02010>`_.
.. [2] A. Miller, A. Glos, and Z. Zimborás, Treespilation: Architecture- and State-Optimised
       Fermion-to-Qubit Mappings (2024), `arXiv:2403.03992 <https://arxiv.org/abs/2403.03992>`_.
