.. _2d_fermi_hubbard:

Simulate 2D Fermi-Hubbard dynamics with flow sets
=================================================

.. meta::
   :description: Tutorial that extends the flow-set approach to the spinless two-dimensional
                 Fermi-Hubbard model on a square lattice with the Verstraete-Cirac encoding, then
                 verifies the circuit, the physics on the stabilizer subspace, and Trotter
                 convergence.

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available in the C API in a future release.

.. seealso::
   **Read the** :ref:`1D flow-set guide <1d_fermi_hubbard>` **first.** This guide is
   its two-dimensional sequel and does not repeat the information developed there: transfer
   and vertex operators, what a flow set is, how :attr:`~.TransferVertexOperator.groups`
   survives into the circuit, and how a custom encoding is wired into the transpiler
   through :class:`.MapperFnEvolutionSynthesis`.

   See the :ref:`transpilation guide <transpilation_explanation>` for the transpiler
   stages referenced throughout, and the :ref:`mappers guide <mappers_explanation>` for the
   general recipe for writing a custom mapper.

The 1D guide ended by mentioning that local encodings are needed when working with two dimensions.
This guide takes that step, for the **two-dimensional Fermi-Hubbard model on a
square lattice**, using the **Verstraete-Cirac (VC) encoding** [2]_.

There are three differences from the 1D problem, forming the basis of this work:

* **The flow sets multiply.** A chain has two arrow orientations; a square lattice has
  four (east, west, north, south), and each is a union of vertex-disjoint directed
  paths (one per lattice row or column) rather than a single path.
* **The encoding gains an ancilla per site**, not one for the whole system, giving a
   qubit-to-mode ratio of :math:`2`. Transfer operators become weight-3 (horizontal) and
   weight-4 (vertical) Paulis, and each of the four flow sets maps to its own Pauli
   shape, so each needs its own diagonalizing Clifford.
* **The encoded Hamiltonian is only faithful on a subspace**: the joint :math:`+1`
   eigenspace of the plaquette constraints that every 2D local encoding carries [2]_ [3]_.
   In 1D the encoding was a plain isometry and the verification was a spectrum check. Here,
   the comparison has to be projected, as :ref:`step 9 <flowsets_2d_stabilizers>` illustrates.

The payoff is the same shape as in 1D: the two-qubit depth of one Trotter step becomes
constant in the lattice size, at :math:`24` (:math:`16` for the hopping, matching
Ref. [1]_, plus :math:`8` for the interaction), while a term-by-term Trotterization of the
same encoded Hamiltonian grows linearly. The gate count still grows with the number
of bonds, of course; what becomes size-independent is how many layers those gates occupy. As in
1D, the
:class:`~qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.CommutingDiagonalizationSynthesis`
plugin finds the circuit of every flow set, here on the qubit pairs that the encoding couples. Mapping the encoding onto a device's
restricted coupling map is its own problem, which is discussed at the end.

The lattice and its four orientations
-------------------------------------

The first change is visible on the directed interaction graph. Sites
are numbered in row-major order, ``site(r, c) = r * cols + c``, and every bond carries both
orientations, as the chain did in 1D, but now the bonds run along two axes:

.. plot::
   :context: close-figs
   :alt: A 3x3 square lattice with both orientations of every horizontal and vertical bond
         drawn as arrows, the center site carrying all four.

   >>> import matplotlib.pyplot as plt
   >>> import numpy as np
   >>> import rustworkx as rx
   >>> from rustworkx.visualization import mpl_draw
   >>>
   >>> rows, cols = 3, 3
   >>> site = lambda r, c: r * cols + c
   >>> graph = rx.PyDiGraph()
   >>> _ = graph.add_nodes_from(range(rows * cols))
   >>> bonds = [(site(r, c), site(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
   >>> bonds += [(site(r, c), site(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
   >>> _ = graph.add_edges_from([(j, k, (j, k)) for j, k in bonds])
   >>> _ = graph.add_edges_from([(k, j, (k, j)) for j, k in bonds])
   >>>
   >>> positions = {site(r, c): (c, -r) for r in range(rows) for c in range(cols)}
   >>> figure, axes = plt.subplots(figsize=(6.6, 5.4), layout="constrained")
   >>> _ = mpl_draw(
   ...     graph,
   ...     pos=positions,
   ...     labels=lambda v: f"$V_{{{v}}}$",
   ...     with_labels=True,
   ...     node_color="orange",
   ...     ax=axes,
   ... )
   >>> # `edge_labels=` would stack both orientations of a horizontal bond on the same
   >>> # point, so place each label beside its own arc instead: the curve bows to the
   >>> # right of j -> k, so offset along that side's normal.
   >>> for j, k in bonds:
   ...     for tail, head in ((j, k), (k, j)):
   ...         (x1, y1), (x2, y2) = positions[tail], positions[head]
   ...         dx, dy = x2 - x1, y2 - y1
   ...         length = np.hypot(dx, dy)
   ...         # the arc's apex sits `arc_rad / 2` from the chord, so clear it by more
   ...         _ = axes.text(
   ...             (x1 + x2) / 2 + dy / length * 0.2,
   ...             (y1 + y2) / 2 - dx / length * 0.2,
   ...             f"$T_{{{tail},{head}}}$",
   ...             ha="center", va="center", fontsize=8,
   ...         )
   >>> axes.set_aspect("equal")
   >>> figure
   <Figure size ... with 1 Axes>

The commutation rule is the 1D one, unchanged: two transfer operators sharing a site
**commute** when the arrows flow through it and **anticommute** when they clash. However,
a site now has up to four incident bonds rather than two, so the
arrows through it can clash in more ways, and a set of arrows that never clashes is no
longer a single path. Reading the horizontal arrows left-to-right gives one such set, the
vertical arrows from top-to-bottom gives another, and reversing each gives two more.
These are the east, west, north, and south flow sets that organize everything that follows.

The Verstraete-Cirac encoding
-----------------------------

Verstraete-Cirac (VC) pairs every fermionic mode :math:`j` with its own auxiliary qubit. With :math:`N`
lattice sites, mode :math:`j` lives on qubit :math:`j` and its ancilla is on qubit
:math:`N + j`, so the register holds :math:`2N` qubits. Vertex operators stay as simple as
they can be,

.. math::

   V_j = Z_j \, ,

which is what makes the occupations directly readable later. Transfer operators depend on
the edge orientation. For a horizontal edge :math:`j \to k` with :math:`j` the western
site, and a vertical edge with :math:`j` the northern site,

.. math::

   T_{jk}^{\text{horizontal}} = \tfrac{1}{2} X_j X_k Z_{N+j} \, , \qquad
   T_{jk}^{\text{vertical}}   = \tfrac{1}{2} X_j Y_k X_{N+j} Y_{N+k} \, ,

a weight-3 "triangle" and a weight-4 "square". Identifying each arrow matters: with
:math:`j` as the western site, :math:`j \to k` points east, but with :math:`j` as the northern
site, it points south. The reverse orientations follow from the identity
:math:`T_{kj} = -V_j V_k T_{jk}`, which turns the ``XXZ`` triangle into ``YYZ``, and
the ``XYXY`` square into ``YXXY``. Those four distinct labels are the four flow
sets, and the rest of this guide tracks them.

.. note::
   **These are not quite Eq. (A1) of Ref.** [1]_, which states "up to single-qubit
   rotations" and differs in three ways worth naming, since a reader checking the paper
   will notice:

   * **The prefactor:** Eq. (A1) carries none. The :math:`\tfrac12` here is what makes
     :math:`T_{jk}^2 = \tfrac14`, matching the normalization
     :math:`T_{jk} = -\tfrac12 (a^\dagger_j - a_j)(a^\dagger_k + a_k)` that
     :class:`.TransferVertexOperator` uses. The paper only ever needs its transfer
     operators inside :math:`\exp(-i\,dt\,H)`, where a scale factor is absorbed into the
     rotation angle; here they have to be *operators* that satisfy an identity.
   * **Orientation:** Eq. (A1) selects letters by the row parity of :math:`j`, which for a
     horizontal edge is the same for both endpoints, so it does not distinguish
     :math:`T_{jk}` from :math:`T_{kj}`. The paper recovers the orientation separately, by
     conjugating with a layer of :math:`R^Z(\pi/2)`. This guide instead folds it into the
     encoding, deriving the reverse from :math:`T_{kj} = -V_j V_k T_{jk}`, which is what
     makes the two orientations land on different Paulis in step 3.
   * **The vertical shape:** Eq. (A1) alternates ``XXXX`` and ``YYYY`` by row parity, where
     this guide uses a single ``XYXY``. These are two ways of getting the same thing: a
     letter asymmetry that makes a vertical edge clash correctly with the horizontal edges
     meeting it. Do not read the difference as a free basis choice. Dropping the parity
     alternation and using a single ``XXXX`` breaks the algebra outright, which is why the
     distinct letters are checked rather than assumed.

.. note::
   The ancilla is attached to a *site*, giving the ratio :math:`2`; it is the
   Derby-Klassen encoding that places an auxiliary qubit per (odd) plaquette, for the
   smaller ratio :math:`3/2`. VC is used here because its transfer operators are the ones
   whose diagonalizing Cliffords Ref. [1]_ gives in closed form.

1. The Hamiltonian on a square lattice
--------------------------------------

The construction is the same as in 1D, but with a two-dimensional edge list. Sites keep the row-major
numbering used for the figure above, ``site(r, c) = r * cols + c``, so that horizontal bonds
connect consecutive indices, and vertical bonds connect indices differing by ``cols``:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from collections import defaultdict
   >>> from qiskit_fermions.operators import TransferVertexOperator
   >>>
   >>> def fermi_hubbard_2d(rows, cols, tunneling, interaction):
   ...     site = lambda r, c: r * cols + c
   ...     edges = [(site(r, c), site(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
   ...     edges += [(site(r, c), site(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
   ...     data: defaultdict[tuple[tuple[int, int], ...], complex] = defaultdict(complex)
   ...     for j, k in edges:
   ...         # hopping: -t (a^dag_j a_k + a^dag_k a_j) = t (T_jk + T_kj)
   ...         data[((j, k),)] += tunneling
   ...         data[((k, j),)] += tunneling
   ...         # interaction: U n_j n_k with n_j = (1 - V_j) / 2
   ...         data[()] += interaction / 4
   ...         data[((j, j),)] -= interaction / 4
   ...         data[((k, k),)] -= interaction / 4
   ...         data[((j, j), (k, k))] += interaction / 4
   ...     return TransferVertexOperator.from_dict(data)
   >>>
   >>> rows, cols = 3, 3
   >>> num_sites = rows * cols
   >>> hamiltonian = fermi_hubbard_2d(rows, cols, tunneling=1.0, interaction=2.0)
   >>> print(f"Number of terms: {len(hamiltonian)}")
   Number of terms: 46

2. Four flow sets instead of two
--------------------------------

Each hopping term is classified by the direction its arrow points on the lattice. Comparing
the ``divmod`` row/column coordinates of the two endpoints separates horizontal from
vertical edges, and the sign of the difference picks the orientation. Everything diagonal
(the identity, the :math:`V_j` and the :math:`V_j V_k` terms) goes into a fifth group:

.. note::
   **Orientation convention.** The names are simply read off the figure above: an arrow is
   **east** or **west** as it points right or left along a row, and **north** or **south** as
   it points up or down a column. Since the numbering runs row-major, east and south are the
   index-increasing directions. Nothing downstream depends on the names (they only label
   groups), but they are easy to confuse, so it is worth fixing them against the
   picture. Ref. [1]_ names its square-lattice flow sets the same way.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> EAST, WEST, NORTH, SOUTH, DIAGONAL = range(5)
   >>>
   >>> def flow_set_groups(operator, cols):
   ...     groups = []
   ...     for terms, _ in operator.iter_terms():
   ...         if len(terms) == 1 and terms[0][0] != terms[0][1]:
   ...             j, k = terms[0]
   ...             row_j, col_j = divmod(j, cols)
   ...             row_k, col_k = divmod(k, cols)
   ...             if row_j == row_k:
   ...                 groups.append(EAST if col_k > col_j else WEST)
   ...             else:
   ...                 groups.append(SOUTH if row_k > row_j else NORTH)
   ...         else:
   ...             groups.append(DIAGONAL)
   ...     return groups
   >>>
   >>> hamiltonian.groups = flow_set_groups(hamiltonian, cols)
   >>>
   >>> names = ["east", "west", "north", "south", "diagonal"]
   >>> for index, name in enumerate(names):
   ...     count = sum(1 for group in hamiltonian.groups if group == index)
   ...     print(f"{name:9s} {count:2d}")
   east       6
   west       6
   north      6
   south      6
   diagonal  22

Within one flow set every arrow either meets another head-to-tail or does not meet it at
all, as in 1D, so a flow set still commutes term-by-term and its evolution
carries no Trotter error. What is new is that a flow set is now disconnected: the east
set of a ``rows x cols`` lattice consists of ``rows`` independent left-to-right paths. That
disconnection is a gift rather than a complication, as step 6 shows.

.. _flowsets_2d_encoding:

3. Encode the operator
----------------------

The encoding is one function that maps a single generalized transfer operator to a Pauli
string, which is handed to :func:`.map_transfer_vertex_generators`. Compared with the 1D version, the
body has more cases: diagonal, horizontal, vertical, and the reverse orientations through
:math:`T_{kj} = -V_j V_k T_{jk}`.

There is one problem. :func:`.map_transfer_vertex_generators` calls back with a generator and
nothing else, while deciding a bond's Pauli string also needs ``cols`` to recover row and column
coordinates from a mode index. Rather than nest the functions to close over it,
:func:`functools.partial` binds it, which keeps both functions at the top level and
independently callable.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from functools import partial
   >>>
   >>> from qiskit.quantum_info import SparseObservable
   >>> from qiskit_fermions.mappers import map_transfer_vertex_generators
   >>>
   >>> def ancilla_qubit(mode, num_sites):
   ...     """The auxiliary qubit paired with fermionic mode `mode`."""
   ...     return num_sites + mode
   >>>
   >>> def vc_action(generator, width, cols):
   ...     """Map one generalized transfer operator to its Verstraete-Cirac Pauli."""
   ...     num_sites = width // 2
   ...     ancilla = partial(ancilla_qubit, num_sites=num_sites)
   ...     j, k = generator
   ...     if j == k:  # vertex operator V_j = Z_j
   ...         return SparseObservable.from_sparse_list([("Z", [j], 1.0)], num_qubits=width)
   ...     row_j, col_j = divmod(j, cols)
   ...     row_k, col_k = divmod(k, cols)
   ...     if row_j == row_k:  # horizontal: weight-3 triangle
   ...         left, right = (j, k) if col_j < col_k else (k, j)
   ...         operator = SparseObservable.from_sparse_list(
   ...             [("XXZ", [left, right, ancilla(left)], 0.5)], num_qubits=width
   ...         )
   ...         forward = (j, k) == (left, right)
   ...     else:  # vertical: weight-4 square
   ...         up, down = (j, k) if row_j < row_k else (k, j)
   ...         operator = SparseObservable.from_sparse_list(
   ...             [("XYXY", [up, down, ancilla(up), ancilla(down)], 0.5)], num_qubits=width
   ...         )
   ...         forward = (j, k) == (up, down)
   ...     if forward:
   ...         return operator
   ...     # reverse orientation: T_kj = -V_j V_k T_jk
   ...     v_j = SparseObservable.from_sparse_list([("Z", [j], 1.0)], num_qubits=width)
   ...     v_k = SparseObservable.from_sparse_list([("Z", [k], 1.0)], num_qubits=width)
   ...     return (-1.0 * v_j.compose(v_k).compose(operator)).simplify()
   >>>
   >>> def vc_encoding(operator, width, cols):
   ...     """Map a TransferVertexOperator to a qubit observable, Verstraete-Cirac style."""
   ...     return map_transfer_vertex_generators(
   ...         operator,
   ...         partial(vc_action, width=width, cols=cols),
   ...         identity=lambda: SparseObservable.identity(width),
   ...         compose=SparseObservable.compose,
   ...     ).simplify()
   >>>
   >>> # bind the lattice width once; the transpiler only ever passes (operator, width)
   >>> encoding = partial(vc_encoding, cols=cols)
   >>> num_qubits = 2 * num_sites
   >>> num_qubits
   18

The point of grouping before mapping can be seen immediately: each flow set encodes to a
single Pauli shape, uniform across the whole set.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> for index, name in enumerate(names):
   ...     observable = encoding(hamiltonian.split_out_groups()[index], num_qubits)
   ...     shapes = sorted({label for label, _, _ in observable.to_sparse_list()})
   ...     print(f"{name:9s} {shapes}")
   east      ['XXZ']
   west      ['YYZ']
   north     ['YXXY']
   south     ['XYXY']
   diagonal  ['', 'Z', 'ZZ']

The four hopping shapes are distinct, so each flow set gets its own Clifford in
:ref:`step 5 <flowsets_2d_synthesis>`. The reverse orientations are different
Paulis, not the same ones relabeled; this is why a 2D flow-set synthesis
needs four circuits, where the 1D synthesis needs one. The plugin finds each of them from the
terms alone.

.. plot::
   :context:
   :nofigs:

   Note the property the encoding actually has to satisfy, and which none of the visible
   output illustrates: the **mixed commutation relations**. Two transfer operators sharing a
   site must commute when the arrows flow through it and anticommute when they clash, and
   :math:`T_{jk}^2 = \tfrac14` must hold for the normalization to be right. Every printed
   Pauli shape above can stay correct while these are violated (the shapes only say which
   letters appear, not whether the parities work out), so a plausible-looking edit to the
   encoding can silently produce a non-fermionic operator.

   Checked on :math:`3 \times 3`, which has an interior site of degree four and vertically
   stacked edges, over both orientations of every bond. This makes the distinct
   letters in ``XYXY`` load-bearing rather than cosmetic: replacing it with ``XXXX`` keeps
   all four printed shapes plausible but fails 32 of these 88 pairs.

   >>> import itertools
   >>> from qiskit.quantum_info import SparseObservable
   >>>
   >>> def obeys_transfer_vertex_algebra(rows, cols):
   ...     """Check the mixed commutation relations over both orientations of every bond."""
   ...     num_qubits = 2 * rows * cols
   ...     encoding = partial(vc_encoding, cols=cols)
   ...     at = lambda r, c: r * cols + c
   ...     bonds = [(at(r, c), at(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
   ...     bonds += [(at(r, c), at(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
   ...     directed = bonds + [(k, j) for j, k in bonds]
   ...
   ...     def encoded(j, k):
   ...         term = TransferVertexOperator.from_dict({((j, k),): 1.0})
   ...         return encoding(term, num_qubits)
   ...
   ...     first = encoded(*bonds[0])
   ...     if first.compose(first).simplify() != SparseObservable.identity(num_qubits) * 0.25:
   ...         return False
   ...
   ...     for (a, b), (c, d) in itertools.combinations(directed, 2):
   ...         if (a, b) == (d, c):  # an edge against its own reverse
   ...             continue
   ...         shared = set((a, b)) & set((c, d))
   ...         if len(shared) != 1:
   ...             continue
   ...         site_ = shared.pop()
   ...         # the arrows flow when one arrives at the shared site and the other leaves it
   ...         flows = (b == site_ and c == site_) or (d == site_ and a == site_)
   ...         left, right = encoded(a, b), encoded(c, d)
   ...         if left.commutes(right) != flows:
   ...             return False
   ...     return True
   >>>
   >>> obeys_transfer_vertex_algebra(3, 3)
   True

.. _flowsets_2d_transpile:

4. Transpile with the custom encoding
-------------------------------------

This stage is the 1D one with a wider qubit register:
:class:`.CustomF2QLayout` maps the :math:`N`-mode register onto :math:`2N` qubits,
:class:`.MapperFnEvolutionSynthesis` carries the encoding, and its ``product_formula``
argument selects how the resulting
:class:`~qiskit.circuit.library.PauliEvolutionGate` is decomposed. Compare
:ref:`step 6 of the 1D guide <flowsets_transpile>`, which builds the same pipeline for a
single ancilla.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit.circuit import QuantumRegister
   >>> from qiskit.passmanager import MultiStagePassManager
   >>> from qiskit.synthesis import LieTrotter
   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import Evolution
   >>> from qiskit_fermions.circuit.library.synthesis import FermionicLieTrotter
   >>> from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
   >>> from qiskit_fermions.transpiler.passes import (
   ...     CustomF2QLayout,
   ...     F2QSynthesis,
   ...     MapperFnEvolutionSynthesis,
   ... )
   >>>
   >>> total_time = 1.0
   >>>
   >>> def flow_set_pass_manager(mode_register, encoding, product_formula=None):
   ...     """Build the pipeline mapping `mode_register` with the VC `encoding`."""
   ...     synthesis = F2QSynthesis()
   ...     synthesis.methods["Evolution"] = MapperFnEvolutionSynthesis(
   ...         encoding, product_formula=product_formula
   ...     )
   ...     # this encoding spends one ancilla per fermionic mode
   ...     qubit_register = QuantumRegister(2 * mode_register.size, "q")
   ...     return MultiStagePassManager(
   ...         input=FermionicCircuitToDAG(),
   ...         layout=CustomF2QLayout({mode_register: qubit_register}),
   ...         synthesis=synthesis,
   ...         output=QuantumDAGToCircuit(),
   ...     )
   >>>
   >>> circuit = FermionicCircuit(num_sites)
   >>> circuit.append(
   ...     Evolution(
   ...         num_sites, hamiltonian, time=total_time, synthesis=FermionicLieTrotter()
   ...     ),
   ...     circuit.modes,
   ... )
   >>>
   >>> # as in 1D, one decompose() splits the Evolution group by group (five gates here,
   >>> # the four hopping flow sets in group order, then the diagonal interaction)
   >>> flow_set_circuit = circuit.decompose()
   >>> flow_set_circuit.count_ops()["Evolution"]
   5

Five gates, carrying the 6, 6, 6, 6, and 22 terms counted in step 2. Each is mapped and
synthesized independently, so the partition decided at the fermionic level survives all the
way into the circuit:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> pass_manager = flow_set_pass_manager(flow_set_circuit.register, encoding, LieTrotter())
   >>> qubit_circuit = pass_manager.run(flow_set_circuit)
   >>> qubit_circuit.num_qubits
   18
   >>>
   >>> def two_qubit_stats(circuit):
   ...     decomposed = circuit.decompose(reps=6)
   ...     two_qubit = lambda instruction: len(instruction.qubits) == 2
   ...     return decomposed.count_ops().get("cx", 0), decomposed.depth(two_qubit)
   >>>
   >>> two_qubit_stats(qubit_circuit)
   (144, 47)

That is the baseline: the encoding and the grouping are both in place, but
:class:`~qiskit.synthesis.LieTrotter` synthesizes each flow set term by term, and the
depth grows with the lattice.

.. _flowsets_2d_synthesis:

5. A shared Clifford per flow set
---------------------------------

The mechanism is the same as in 1D: conjugate a whole commuting flow set by one Clifford, so
that the evolution of the entire set costs little more than single-qubit rotations. In 1D, one
brickwork of ``CZ`` gates did this, and
:class:`~qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.CommutingDiagonalizationSynthesis`
found it on its own. It does so here too, and the only thing it needs to know about the device is which pairs
of qubits can interact.

Ref. [1]_ builds the Cliffords of the four flow sets from two layers of ``CX`` gates on one line
of the lattice at a time (a row for east and west, a column for north and south) together with
that line's ancillas (Section IV C and Fig. 7). Its gates act on four kinds of pairs:

- A site and its own ancilla.
- Two neighboring sites.
- Two neighboring ancillas.
- Along a row, a site and the ancilla of its left neighbor.

Those are the pairs this guide gives the plugin, in the qubit indices of the transpiled circuit
(the sites first, then their ancillas). Every two-qubit gate the plugin then emits is on one of
them, so the circuit needs no routing on a device that couples them:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> def register_pairs(rows, cols):
   ...     """The qubit pairs of the VC register that the flow-set Cliffords act on."""
   ...     num_sites = rows * cols
   ...     ancilla = partial(ancilla_qubit, num_sites=num_sites)
   ...     pairs = [(j, ancilla(j)) for j in range(num_sites)]
   ...     for j in range(num_sites):
   ...         row, col = divmod(j, cols)
   ...         if col + 1 < cols:  # the right neighbor in the row
   ...             pairs += [(j, j + 1), (ancilla(j), ancilla(j + 1)), (j + 1, ancilla(j))]
   ...         if row + 1 < rows:  # the neighbor below in the column
   ...             pairs += [(j, j + cols), (ancilla(j), ancilla(j + cols))]
   ...     return pairs

The plugin offers its device-aware candidates only when it knows the allowed pairs: from the
device once the gate's qubits are placed on it, or from ``edges``, as here. It ranks every
candidate circuit by its two-qubit depth, then its two-qubit gate count. For the vertical flow
sets, its cheapest circuit is a layer of ``CZ`` gates, a layer of generic two-qubit blocks (two
``CX`` gates each), and the ``CZ`` layer again. The ``CZ`` layer only has to bring every term onto
one allowed pair, which is cheaper than diagonalizing every term, and each block carries the
evolution of the terms on its pair. Drawn for the north flow set (the third gate of the qubit
circuit of step 4) on the :math:`3 \times 3` lattice built in step 1:

.. plot::
   :context: close-figs
   :include-source:
   :alt: The plugin's circuit for the north flow set on a 3 by 3 lattice: a layer of CZ gates,
         a layer of two-qubit unitary blocks, and the same CZ layer again, with single-qubit
         gates in between.

   >>> from qiskit_fermions.transpiler.passes import CommutingDiagonalizationSynthesis
   >>>
   >>> plugin = CommutingDiagonalizationSynthesis(edges=register_pairs(rows, cols))
   >>> north = qubit_circuit.data[2].operation
   >>> plugin.run(north).draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

The lines of a flow set act on disjoint qubits, so their ``CZ`` layers and blocks run side by
side, and adding lines to the lattice widens the circuit without deepening it. That is the whole
constant-depth claim. This flow set costs four two-qubit layers: one for each ``CZ`` layer and
two for the blocks. The other three cost four as well, whatever circuit the plugin picks for them.

.. plot::
   :context:
   :nofigs:

   The constant-depth result rests on two claims that the drawing shows only for one flow set
   on one lattice: every hopping flow set costs four two-qubit layers whatever the lattice size,
   and every two-qubit gate is on an allowed pair. Both are swept here over the four hopping
   flow sets and several lattice sizes, rectangular ones included, so that a confusion between
   rows and columns cannot cancel out.

   >>> from qiskit.circuit.library import PauliEvolutionGate
   >>>
   >>> def hopping_sets_are_local_and_shallow(rows, cols):
   ...     """Every hopping flow set: two-qubit depth 4, all two-qubit gates on allowed pairs."""
   ...     pairs = {tuple(sorted(pair)) for pair in register_pairs(rows, cols)}
   ...     encoding = partial(vc_encoding, cols=cols)
   ...     operator = fermi_hubbard_2d(rows, cols, 1.0, 2.0)
   ...     operator.groups = flow_set_groups(operator, cols)
   ...     plugin = CommutingDiagonalizationSynthesis(edges=list(pairs))
   ...     for group in operator.split_out_groups()[:4]:  # the hopping sets come first
   ...         gate = PauliEvolutionGate(encoding(group, 2 * rows * cols), time=0.3)
   ...         circuit = plugin.run(gate).decompose(reps=6)
   ...         used = {
   ...             tuple(sorted(circuit.find_bit(q).index for q in inst.qubits))
   ...             for inst in circuit.data
   ...             if len(inst.qubits) == 2
   ...         }
   ...         if circuit.depth(lambda inst: len(inst.qubits) == 2) != 4 or not used <= pairs:
   ...             return False
   ...     return True
   >>>
   >>> all(hopping_sets_are_local_and_shallow(r, c) for r, c in [(2, 3), (3, 2), (4, 4), (5, 5)])
   True

6. Flow-set-aware synthesis
---------------------------

The synthesis gets a helpful head start. Because :class:`.Evolution` splits itself group by
group, the high-level synthesis hands the plugin one flow set per gate (five times here, four
hopping sets plus the diagonal group), so it never has to re-derive the partition, and every
gate it sees consists of commuting terms. Then comes the observation that makes two dimensions
cheaper than one might fear:

.. important::
   The lines of a single flow set are vertex-disjoint: the west set's rows share no qubits.
   The plugin solves the disjoint parts of a gate one by one and emits them side by side, so
   all the lines of a flow set share the same two-qubit layers, rather than queueing up one
   line after another. Lines that are copies of each other are even solved once. A flow set
   therefore costs a fixed number of layers no matter how many rows or columns the lattice
   has, which is why the total depth ends up constant.

As in 1D, a :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass applies the plugin to the
qubit circuit. Here it is given as an instance, to pass it the pairs of the register:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit.transpiler import PassManager
   >>> from qiskit.transpiler.passes import HighLevelSynthesis
   >>> from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
   >>>
   >>> def commuting_synthesis(rows, cols):
   ...     """Synthesize every flow set with the plugin, on the pairs of the VC register."""
   ...     plugin = CommutingDiagonalizationSynthesis(edges=register_pairs(rows, cols))
   ...     return PassManager(HighLevelSynthesis(hls_config=HLSConfig(PauliEvolution=[plugin])))

.. note::
   As in 1D, the plugin takes no number of Trotter steps. A Trotter step has to cycle *between*
   the flow sets, but ``transpile_trotter_step`` below decomposes the fermionic circuit into one
   :class:`.Evolution` gate per set, so the plugin only ever sees one set at a time. It
   synthesizes that set without approximation, and a repetition *within* it would change
   nothing: repeating one operator just splits its own evolution into equal pieces.

   Multi-step Trotterization therefore needs a product formula that respects the operator's
   :attr:`~qiskit_fermions.operators.OperatorTrait.groups` inside a single gate, before the
   decomposition. :class:`.FermionicLieTrotter` does that, and
   :ref:`step 10 <flowsets_2d_convergence>` uses it to measure the convergence.

   Note the two similarly named classes at play. Qiskit's
   :class:`~qiskit.synthesis.LieTrotter`, used as the baseline above, acts on the *mapped*
   Hamiltonian, where the grouping is already gone. :class:`.FermionicLieTrotter` is the fermionic
   counterpart, and it is the one that can still see the flow sets.

7. The result: constant depth
-----------------------------

Everything so far ran on the single :math:`3 \times 3` lattice built in step 1. Comparing the
two synthesis paths across growing lattices needs that whole sequence (operator,
grouping, encoding, circuit, pass manager) repeated per size, so it is worth wrapping once:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> def transpile_trotter_step(rows, cols, product_formula=None, steps=1):
   ...     """Transpile `steps` Trotter steps on a `rows` x `cols` lattice.
   ...
   ...     Every flow set is synthesized by the plugin, unless a qubit-side product formula is
   ...     given for the baseline.
   ...     """
   ...     num_sites = rows * cols
   ...     operator = fermi_hubbard_2d(rows, cols, 1.0, 2.0)
   ...     operator.groups = flow_set_groups(operator, cols)
   ...     encoding = partial(vc_encoding, cols=cols)
   ...
   ...     circuit = FermionicCircuit(num_sites)
   ...     circuit.append(
   ...         Evolution(
   ...             num_sites,
   ...             operator,
   ...             time=total_time,
   ...             synthesis=FermionicLieTrotter(reps=steps),
   ...         ),
   ...         circuit.modes,
   ...     )
   ...     circuit = circuit.decompose()  # split into one Evolution per flow set
   ...
   ...     pass_manager = flow_set_pass_manager(circuit.register, encoding, product_formula)
   ...     qubit_circuit = pass_manager.run(circuit)
   ...     if product_formula is not None:
   ...         return qubit_circuit
   ...     return commuting_synthesis(rows, cols).run(qubit_circuit)

Collecting both metrics across a range of lattice sizes and plotting them against the lattice
edge makes the difference plain:

.. plot::
   :context: close-figs
   :include-source:
   :alt: Two-qubit depth and CX count against lattice edge; the qubit-level LieTrotter grows
         linearly while the flow-set synthesis stays flat at 24.

   >>> sizes = [3, 4, 5, 6, 7, 8]
   >>> lie_trotter = [
   ...     two_qubit_stats(transpile_trotter_step(n, n, LieTrotter())) for n in sizes
   ... ]
   >>> flow = [two_qubit_stats(transpile_trotter_step(n, n)) for n in sizes]
   >>>
   >>> figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
   >>> for panel, index, title in zip(axes, (1, 0), ("two-qubit depth", "CX count")):
   ...     _ = panel.plot(sizes, [stat[index] for stat in lie_trotter], "o-", label="qubit-level")
   ...     _ = panel.plot(sizes, [stat[index] for stat in flow], "s-", label="flow set")
   ...     _ = panel.set(xlabel="lattice edge $L$ (an $L \\times L$ lattice)", title=title)
   ...     _ = panel.legend()
   >>> figure
   <Figure size ... with 2 Axes>

The two-qubit depth is 24 and constant from :math:`3 \times 3` to :math:`8 \times 8`
(that is 18 qubits up to 128), while :class:`~qiskit.synthesis.LieTrotter` applied to
the identically encoded Hamiltonian grows linearly with the lattice edge, from 47 to 92. The
gate count grows under both, since there are :math:`O(N)` terms to apply either way; what
the flow sets remove is the growth in how many layers those gates occupy.

The budget splits into 16 for the hopping and eight for the interaction:

* Each of the four flow sets costs four two-qubit layers, as step 5 checked; in the circuit drawn
  there, a ``CZ`` layer going in, the two-qubit blocks, and the ``CZ`` layer coming out. The
  circuits of Ref. [1]_ spend the same four layers differently, on a depth-two Clifford going in
  and its inverse coming out, with only single-qubit rotations in between. Either way that is
  :math:`4 \times 4 = 16`, which is the figure Ref. [1]_ quotes, since it counts the hopping terms
  only. Setting :math:`U = 0` in the code above reproduces it.
* The diagonal group adds the remaining eight. Its ``Rzz`` gates pack into four layers, and each
  ``Rzz`` decomposes into two ``CX`` gates around a single-qubit rotation, so :math:`4 \times 2 = 8`.
  Four layers is optimal: on a square lattice, the interaction reaches only nearest
  neighbors, so every interior site carries four ``Rzz`` gates (one per bond) and can
  take part in only one at a time.

Nothing in either sum refers to the lattice size, which is the whole point.

.. _flowsets_2d_verify:

8. Verify the circuit
---------------------

The flow-set synthesis and :class:`~qiskit.synthesis.LieTrotter` describe the same Trotter
step of the same encoded Hamiltonian, so their unitaries must agree up to a global phase.
Random statevectors detect any discrepancy far more cheaply than building the full unitary:

.. note::
   Both circuits are decomposed down to one- and two-qubit gates *before* being applied to a
   state. Left un-decomposed, each high-level ``Evolution`` instruction forces
   :class:`~qiskit.quantum_info.Statevector` to build its full :math:`2^{2N} \times 2^{2N}`
   dense operator; decomposing first lets it apply gates one at a time instead, which is
   several hundred times faster here.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import numpy as np
   >>> from qiskit.quantum_info import Statevector
   >>>
   >>> rng = np.random.default_rng(1234)
   >>>
   >>> def agrees_up_to_one_phase(rows, cols, samples=3):
   ...     """Compare both syntheses on random states.
   ...
   ...     Returns whether they agree up to a phase, and whether it is the *same* phase
   ...     for every input state.
   ...     """
   ...     lie_trotter = transpile_trotter_step(
   ...         rows, cols, LieTrotter()
   ...     ).decompose(reps=6)
   ...     flow = transpile_trotter_step(rows, cols).decompose(reps=6)
   ...     dimension = 2**lie_trotter.num_qubits
   ...     residuals, phases = [], []
   ...     for _ in range(samples):
   ...         vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
   ...         state = Statevector(vector / np.linalg.norm(vector))
   ...         a = state.evolve(lie_trotter).data
   ...         b = state.evolve(flow).data
   ...         overlap = np.vdot(a, b)
   ...         phases.append(overlap / abs(overlap))
   ...         residuals.append(np.linalg.norm(b - a * overlap / abs(overlap)))
   ...     # the SAME global phase must work for every input state
   ...     spread = max(abs(phase - phases[0]) for phase in phases)
   ...     return bool(max(residuals) < 1e-9), bool(spread < 1e-9)
   >>>
   >>> agrees_up_to_one_phase(rows, cols)
   (True, True)

.. note::
   Both halves of that check matter. A unit-modulus overlap on a single state is necessary
   but not sufficient: a synthesis that got a state-dependent phase wrong would still
   show it. Requiring the same phase across several random states is what closes the gap.

.. plot::
   :context:
   :nofigs:

   The visible check above runs on a square lattice, where a row line and a column line have
   the same length. That symmetry hides a whole class of mistake: an index confusion between
   ``rows`` and ``cols``, or between a row line and a column line, can cancel out on
   :math:`3 \times 3` and still be wrong everywhere else. Repeat the check on the two
   rectangular lattices, which are transposes of each other, so the row and column paths are
   exercised at different lengths.

   >>> all(agrees_up_to_one_phase(r, c) == (True, True) for r, c in [(2, 3), (3, 2)])
   True


.. _flowsets_2d_stabilizers:

9. Verify the physics on the stabilizer subspace
------------------------------------------------

This is the two-dimensional wrinkle. In 1D, the encoding was an isometry, and
comparing spectra against a Jordan-Wigner reference was enough. VC is not an isometry: it
uses :math:`2N` qubits for :math:`N` modes, so a :math:`2^{2N}`-dimensional space holds a
:math:`2^N`-dimensional physical one. This is generic to local encodings in two dimensions
(the surplus qubits are constrained by local commuting symmetries, one per plaquette
[2]_ [3]_) and it means the encoded Hamiltonian represents the fermionic one faithfully
only on the joint :math:`+1` eigenspace of those constraints.

.. warning::
   The word **stabilizer** does double duty in this subject, and the two meanings are
   unrelated. Ref. [1]_ uses it for the commuting transfer operators of a flow set, which
   form a stabilizer group that the Cliffords of :ref:`step 5 <flowsets_2d_synthesis>`
   diagonalize (a compilation device, and the sense meant everywhere above). This section
   uses it in the error-correction sense: operators whose :math:`+1` eigenspace is the
   physical subspace. Ref. [1]_ explicitly flags this collision and sets the second meaning
   aside; the plaquette constraints below are a property of the VC encoding [2]_, not a
   result of the flow-set framework.

The generators are constructed here rather than quoted: one per square face, and each is
simply the encoded **edge**-operator loop around that face.

Edge operators are the natural object for this. A closed loop of them is a symmetry of any
particle-number-conserving Hamiltonian, whereas the corresponding loop of transfer operators
is not: the two differ by a vertex operator on each bond's source site,

.. math::

   T_{jk} = \tfrac{i}{2} \, V_j E_{jk}
   \qquad \Longleftrightarrow \qquad
   E_{jk} = -2i \, V_j T_{jk} \, ,

Around a closed loop every site is the source of exactly one bond, so the loop collects one
:math:`V_j` per site, which is the **local parity** of the plaquette's four sites. That
local parity is what makes the difference: the Hamiltonian hops particles across the plaquette
boundary, so it conserves only the *global* parity. The bare transfer loop therefore fails to commute with it, while the edge loop (which
carries the parity factor) commutes.

Since the encoding is defined on transfer operators, mapping an edge operator needs the relation
above. That is a five-line callback, and it slots into
:func:`.map_edge_vertex_generators` exactly as ``vc_action`` slots into
:func:`.map_transfer_vertex_generators`:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.mappers import map_edge_vertex_generators
   >>> from qiskit_fermions.operators import EdgeVertexOperator
   >>>
   >>> def vc_edge_action(generator, width, cols):
   ...     """Map one generalized *edge* operator to its Verstraete-Cirac Pauli."""
   ...     j, k = generator
   ...     if j == k:  # a vertex operator is the same object in both algebras
   ...         return SparseObservable.from_sparse_list([("Z", [j], 1.0)], num_qubits=width)
   ...     # The library decomposes T_jk = (i/2) V_j E_jk, so E_jk = -2i V_j T_jk. Reuse the
   ...     # transfer encoding for T_jk rather than repeating its four cases here. Note that
   ...     # `compose` left-multiplies its argument, so this composes to V_j T_jk; the other
   ...     # order carries the opposite sign, since a vertex and a transfer operator sharing a
   ...     # mode anticommute.
   ...     vertex = SparseObservable.from_sparse_list([("Z", [j], 1.0)], num_qubits=width)
   ...     return (-2.0j * vc_action(generator, width, cols).compose(vertex)).simplify()
   >>>
   >>> def vc_edge_encoding(operator, width, cols):
   ...     """Map an EdgeVertexOperator to a qubit observable, Verstraete-Cirac style."""
   ...     return map_edge_vertex_generators(
   ...         operator,
   ...         partial(vc_edge_action, width=width, cols=cols),
   ...         identity=lambda: SparseObservable.identity(width),
   ...         compose=SparseObservable.compose,
   ...     ).simplify()
   >>>
   >>> def plaquette_loops(rows, cols):
   ...     """The directed edge loop around each square face."""
   ...     at = lambda r, c: r * cols + c
   ...     return [
   ...         [(at(r, c), at(r, c + 1)), (at(r, c + 1), at(r + 1, c + 1)),
   ...          (at(r + 1, c + 1), at(r + 1, c)), (at(r + 1, c), at(r, c))]
   ...         for r in range(rows - 1)
   ...         for c in range(cols - 1)
   ...     ]
   >>>
   >>> def stabilizers(rows, cols):
   ...     """One plaquette stabilizer per square face of the lattice."""
   ...     num_qubits = 2 * rows * cols
   ...     out = []
   ...     for loop in plaquette_loops(rows, cols):
   ...         loop_operator = EdgeVertexOperator.from_dict({tuple(loop): 1.0})
   ...         encoded_loop = vc_edge_encoding(loop_operator, num_qubits, cols)
   ...         out.append(encoded_loop / complex(encoded_loop.coeffs[0]))
   ...     return out
   >>>
   >>> plaquette_stabilizers = stabilizers(rows, cols)
   >>> for stabilizer in plaquette_stabilizers:
   ...     print(stabilizer.to_sparse_list())
   [('ZZYXXY', [1, 4, 9, 10, 12, 13], (1+0j))]
   [('ZZYXXY', [2, 5, 10, 11, 13, 14], (1+0j))]
   [('ZZYXXY', [4, 7, 12, 13, 15, 16], (1+0j))]
   [('ZZYXXY', [5, 8, 13, 14, 16, 17], (1+0j))]

.. plot::
   :context:
   :nofigs:

   Two claims underpin the construction above, and neither is visible in the printed Paulis.

   First, the edge loop is the right object: the *transfer* loop on its own does **not**
   commute with the encoded Hamiltonian, and the bonds that spoil it are exactly the ones
   leaving the plaquette. The two loops differ by the local parity of the plaquette's sites,
   and the Hamiltonian conserves only the global parity, so the local factor is what repairs
   the commutation.

   Second, every stabilizer acts *diagonally* on the site qubits (only ``Z`` appears on a site
   qubit, with the ``X``/``Y`` content confined to the ancillas). That is what lets a site occupation
   be written as a plain basis state and survive the projection below unchanged: a diagonal
   operator cannot move it. An encoding whose plaquette operators reached the sites with
   ``X``/``Y`` would silently scramble the physical state instead.

   >>> encoded_hamiltonian = encoding(hamiltonian, num_qubits)
   >>>
   >>> transfer_loops = [
   ...     encoding(TransferVertexOperator.from_dict({tuple(loop): 1.0}), num_qubits)
   ...     for loop in plaquette_loops(rows, cols)
   ...     ]
   >>> [loop.commutes(encoded_hamiltonian) for loop in transfer_loops]
   [False, False, False, False]
   >>> [s.commutes(encoded_hamiltonian) for s in plaquette_stabilizers]
   [True, True, True, True]

   >>> def site_restriction(stabilizer, num_sites):
   ...     """The stabilizer's Pauli letters on the site qubits only."""
   ...     return {
   ...         letter
   ...         for label, indices, _ in stabilizer.to_sparse_list()
   ...         for letter, qubit in zip(label, indices)
   ...         if qubit < num_sites
   ...     }
   >>>
   >>> all(site_restriction(s, num_sites) <= {"Z"} for s in plaquette_stabilizers)
   True

Each generator has weight :math:`6`, spread over two of its plaquette's physical qubits and
four ancillas. They are involutory and commute with the encoded Hamiltonian, so the physical
subspace is preserved by the dynamics. Both properties are statements about operators, so
they are checked symbolically on the :class:`~qiskit.quantum_info.SparseObservable` algebra,
no :math:`2^{18} \times 2^{18}` matrices required:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> encoded = encoding(hamiltonian, num_qubits)
   >>> identity = SparseObservable.identity(num_qubits)
   >>> for stabilizer in plaquette_stabilizers:
   ...     square = stabilizer.compose(stabilizer).simplify()
   ...     print(square == identity, stabilizer.commutes(encoded))
   True True
   True True
   True True
   True True

With the stabilizers in hand, the comparison needs no basis correspondence, because
:math:`V_j = Z_j` means the occupations live on the same physical qubits in both
pictures. So: write a product state directly on the physical qubits with the ancillas in
:math:`|0\rangle`, project it onto the stabilizer subspace, evolve with the encoded
Hamiltonian, and read the site densities :math:`\langle n_j \rangle = (1 - \langle Z_j \rangle)/2` off both sides.

Every step here stays at the level of a vector. The projector is applied one
:math:`(1 + S)/2` factor at a time instead of being multiplied out, and the time evolution
uses :func:`~scipy.sparse.linalg.expm_multiply`, which computes
:math:`e^{-iHt} |\psi\rangle` without ever forming :math:`e^{-iHt}`. Since the encoded
Hamiltonian is a sparse :math:`2^{2N}`-dimensional operator, that distinction is what keeps
this section cheap.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit.quantum_info import SparsePauliOp
   >>> from scipy.sparse.linalg import expm_multiply
   >>> from qiskit_fermions.mappers.library import transfer_vertex_jordan_wigner
   >>>
   >>> # an exact evolution time, unrelated to the Trotter step above
   >>> evolution_time = 0.6
   >>> reference = transfer_vertex_jordan_wigner(hamiltonian, num_sites).simplify()
   >>>
   >>> def to_sparse_matrix(observable):
   ...     # SparseObservable carries no dense- or sparse-matrix method, so route through
   ...     # SparsePauliOp. This is only ever needed for the numerical checks below --
   ...     # the encoding itself never leaves SparseObservable.
   ...     return SparsePauliOp.from_sparse_observable(observable).to_matrix(sparse=True)
   >>>
   >>> def basis_state(occupation, num_qubits):
   ...     """The computational basis state with `occupation` on the first qubits.
   ...
   ...     Qubit `q` carries bit `2**q`, and the ancillas are the *high* qubits `N..2N-1`,
   ...     so the same integer indexes the same site occupations on either register and
   ...     leaves every ancilla in |0>.
   ...     """
   ...     index = int("".join(str(bit) for bit in reversed(occupation)), 2)
   ...     state = np.zeros(2**num_qubits, dtype=complex)
   ...     state[index] = 1.0
   ...     return state
   >>>
   >>> def project_onto_codespace(state, stabilizers):
   ...     """Apply the projector one (1 + S) factor at a time, then renormalize.
   ...
   ...     The factors commute, so their product is itself a projector and the order does
   ...     not matter. Multiplying it out would build a dense 2^2N x 2^2N operator; applied
   ...     factor by factor this stays a handful of sparse matrix-vector products.
   ...
   ...     Each factor is really (1 + S)/2, but every one contributes the same scalar, so the
   ...     halvings only rescale the result and the final normalization absorbs them.
   ...     """
   ...     for stabilizer in stabilizers:
   ...         state += to_sparse_matrix(stabilizer) @ state
   ...     return state / np.linalg.norm(state)
   >>>
   >>> occupation = [1, 0, 1, 0, 1, 0, 0, 1, 0]  # site occupations of the initial state
   >>> initial_reference = basis_state(occupation, num_sites)
   >>> initial_encoded = project_onto_codespace(
   ...     basis_state(occupation, num_qubits), plaquette_stabilizers
   ... )
   >>>
   >>> evolved_encoded = expm_multiply(
   ...     -1j * evolution_time * to_sparse_matrix(encoded), initial_encoded
   ... )
   >>> evolved_reference = expm_multiply(
   ...     -1j * evolution_time * to_sparse_matrix(reference), initial_reference
   ... )
   >>>
   >>> def density(state, qubit, width):
   ...     """Site density <n_j> = (1 - <Z_j>) / 2 of a statevector."""
   ...     z = to_sparse_matrix(
   ...         SparseObservable.from_sparse_list([("Z", [qubit], 1.0)], num_qubits=width)
   ...     )
   ...     return (1 - (state.conj() @ (z @ state)).real) / 2
   >>>
   >>> encoded_densities = [density(evolved_encoded, j, num_qubits) for j in range(num_sites)]
   >>> reference_densities = [density(evolved_reference, j, num_sites) for j in range(num_sites)]
   >>>
   >>> for j, density_value in enumerate(encoded_densities):
   ...     print(f"site {j}: {density_value:.10f}")
   site 0: 0.6750993356
   site 1: 0.3689791258
   site 2: 0.6750993356
   site 3: 0.3818130396
   site 4: 0.4868585996
   site 5: 0.3818130396
   site 6: 0.2693544696
   site 7: 0.4916285849
   site 8: 0.2693544696
   >>> np.allclose(encoded_densities, reference_densities, atol=1e-12)
   True

Those are the encoded densities; they match the Jordan-Wigner reference to machine
precision, on a lattice whose encoding is faithful only after projection.

.. _flowsets_2d_convergence:

10. Measure the Trotter convergence
-----------------------------------

:ref:`Step 8 <flowsets_2d_verify>` established that the two syntheses agree with each other, which
is a statement about equivalence rather than accuracy: both describe the same single Trotter step,
so both carry the same Trotter error. Measuring that error needs an exact reference, and step 9 just
built one.

The step count comes from :class:`.FermionicLieTrotter`, the fermionic counterpart of the
qubit-level formula used as the baseline earlier. Applied to the whole grouped operator before the
decomposition, its repetition cycles between the four flow sets and the diagonal group instead of
within any one of them:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> initial_state = project_onto_codespace(
   ...     basis_state(occupation, num_qubits), plaquette_stabilizers
   ... )
   >>> exact = expm_multiply(
   ...     -1j * total_time * to_sparse_matrix(encoded), initial_state
   ... )
   >>>
   >>> def trotter_fidelity(steps):
   ...     # `steps` goes to the fermionic formula inside transpile_trotter_step, so the
   ...     # repetition cycles between the flow sets rather than within any one of them
   ...     circuit = transpile_trotter_step(rows, cols, steps=steps)
   ...     evolved = Statevector(initial_state).evolve(circuit.decompose(reps=6))
   ...     return abs(np.vdot(evolved.data, exact)) ** 2
   >>>
   >>> for steps in (1, 2, 4, 8):
   ...     print(f"{steps} step(s): fidelity {trotter_fidelity(steps):.6f}")
   1 step(s): fidelity 0.053820
   2 step(s): fidelity 0.704014
   4 step(s): fidelity 0.932937
   8 step(s): fidelity 0.982687

The fidelity climbs monotonically toward one as the steps shorten, which is the behavior a product
formula must show. Shortening the steps is only one of the two knobs;
:class:`.FermionicSuzukiTrotter` trades them against each other at a higher order, as
:ref:`fermionic_synthesis_explanation` measures.

A single step of this Hamiltonian at :math:`t = 1` is a poor approximation
(the four hopping sets and the interaction do not commute) so the first row is far from one, and
that is the point: the convergence is what confirms the construction, not any single row.

What to take away
-----------------

- **Two dimensions changes the bookkeeping, not the recipe.** Every ingredient from the 1D
  guide carries over unmodified: :attr:`~.TransferVertexOperator.groups` for the grouping,
  :func:`.map_transfer_vertex_generators` for the encoding,
  :class:`.CustomF2QLayout` for the mismatched register, and
  :class:`.MapperFnEvolutionSynthesis` to wire it into the transpiler, and
  :class:`~qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.CommutingDiagonalizationSynthesis`
  to synthesize each flow set. What grew is the number of cases inside each, and the plugin handles those without being told about them.
- **Disjointness is what makes 2D constant-depth.** A flow set decomposes into
  vertex-disjoint lines, so the circuits of its lines run in parallel and share the same
  two-qubit layers no matter how large the lattice is. Four flow sets and a diagonal group then
  give a fixed budget: here :math:`16` for the four flow sets plus :math:`8` for the diagonal
  interaction group.
- **Each orientation is its own Pauli shape.** :math:`T_{kj} = -V_j V_k T_{jk}` turns
  ``XXZ`` into ``YYZ`` and ``XYXY`` into ``YXXY``, so the four flow sets need four
  Cliffords. Deriving them by hand is where a 2D flow-set synthesis most easily goes wrong; the
  plugin derives each from the terms of its gate, and checks that it sends every term where the
  middle part of the circuit expects it.
- **Name the qubit pairs, not the circuit.** The plugin needs the pairs of qubits that can
  interact, and keeps every two-qubit gate on them. Here those are the pairs of the VC register;
  once a circuit is placed on a device, the plugin reads them off the device's coupling map.
- **Let the grouping do the sorting.** Because :class:`.Evolution` splits itself per group,
  the plugin is handed one flow set at a time and never has to re-derive the partition.
- **The step count sits above the split.** The synthesis here sees one flow set per gate, so it
  cannot subdivide the evolution across the sets. :class:`.FermionicLieTrotter` does that in
  fermionic space, before the decomposition, which is what makes a convergence test possible at all.
- **Verify on the subspace.** A :math:`2N`-qubit encoding of :math:`N` modes is faithful
  only where the stabilizers say it is. Because VC keeps :math:`V_j = Z_j`, the cheapest
  reliable check is to project a product state and compare densities: no intertwiner and
  no spectral matching, which is unreliable when the restricted Hamiltonian is
  degenerate.

.. note::
   The encoded Hamiltonian commutes with the stabilizers, so the projection only has to be
   applied to the initial state. The evolution preserves the subspace by itself. On real
   hardware this is also what makes the encoding useful as an error-detection resource:
   the stabilizers can be measured.

Next
----

Two directions extend what is above, and both are changes of ingredient rather than of
method. The first is the encoding: Verstraete-Cirac was chosen here because Ref. [1]_ gives
its diagonalizing Cliffords in closed form, which makes the plugin's result easy to compare
against, but Derby-Klassen spends one ancilla per odd plaquette instead of one per site, for a
qubit-to-mode ratio of :math:`3/2` rather than :math:`2`. Its transfer operators have different
Pauli shapes, so a port amounts to rewriting the encoding function of
:ref:`step 3 <flowsets_2d_encoding>` and the qubit pairs of
:ref:`step 5 <flowsets_2d_synthesis>`; the plugin finds the Cliffords for the new shapes, and the
grouping, the layout, and the pipeline carry over untouched.

The second is the lattice itself. Everything here assumes open boundaries and a single
fermionic species. Periodic boundaries close each line into a cycle: the arrows still all
commute, since each meets the next head-to-tail, so every flow set is still a group of commuting
terms that the plugin accepts. A cycle of :math:`L` sites carries :math:`L` bonds per
orientation where a line carries :math:`L - 1`, and the wrap-around bond adds pairs of its own,
so whether a flow set still costs four layers on a cycle is a question this guide leaves open. A
spinful model is easier: it doubles the register and adds on-site interaction terms to the
diagonal group, without touching the flow sets. Neither disturbs the constant-depth
argument, but both change the bookkeeping, which, as the first takeaway says, is the part
that two dimensions makes bigger.

.. important::
   **Hardware connectivity is only partly addressed here.** Every two-qubit gate of a Trotter
   step above is on one of the qubit pairs of :ref:`step 5 <flowsets_2d_synthesis>`, so on a
   device that couples those pairs the circuit needs no routing, and the reported depths are its
   depths there. Real devices have a restricted coupling map, and mapping this encoding onto one
   is a substantial problem in its own right: a layout has to keep all four kinds of pairs
   adjacent simultaneously. Where a device couples fewer pairs, give the plugin those (or run the
   high-level synthesis after the layout stage, so that it reads them off the device), and it
   keeps every gate on them, at whatever depth they allow. Otherwise the transpiler inserts
   ``SWAP`` gates, and the layers they add are what the constant-depth construction was buying
   back. Ref. [1]_ likewise quotes its depth figure under an assumed square connectivity graph
   with ``CX`` as the only native entangler. Finding good layouts and routing strategies for
   local encodings on a given device topology is beyond this guide's scope.

.. [1] A. Gandon, S. Piccinelli, M. Rossmannek, F. Tacchino, A. Baiardi, J. Nys, and
       I. Tavernelli, Stabilizer-based quantum simulation of fermion dynamics with local
       qubit encodings (2026), `arXiv:2512.11418v2 <https://arxiv.org/abs/2512.11418v2>`_.

.. [2] F. Verstraete and J. I. Cirac, Mapping local Hamiltonians of fermions to local
       Hamiltonians of spins, `J. Stat. Mech. 2005, P09012
       <https://doi.org/10.1088/1742-5468/2005/09/P09012>`_ (2005).

.. [3] Y.-A. Chen and Y. Xu, Equivalence between fermion-to-qubit mappings in two spatial
       dimensions, `PRX Quantum 4, 010326
       <https://doi.org/10.1103/PRXQuantum.4.010326>`_ (2023).
