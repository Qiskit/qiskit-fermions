.. _flowsets_getting_started:

Simulate 1D Fermi-Hubbard dynamics with flow sets
=================================================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available via the C API in a future release.

This guide builds a circuit for the **time dynamics of the one-dimensional Fermi-Hubbard
model**, from writing down the Hamiltonian to checking the evolved site densities. What
makes the route interesting is a single idea, taken from the *flow set* framework of
Gandon et al. [1]_ --- the work that also motivated the :class:`.TransferVertexOperator`
in this package:

   Instead of grouping Hamiltonian terms *after* the fermion-to-qubit mapping, group
   them *before* it, into **flow sets** -- one-dimensional subsets of the directed
   fermionic interaction graph whose transfer operators mutually commute.

That choice drives everything below. It calls for a **custom fermion-to-qubit encoding**
tailored to the flow sets, which here spends one **ancilla qubit** so that the qubit count
no longer matches the number of fermionic modes, and for a **custom synthesis** that
exploits the commutativity within each set. The payoff is a circuit whose two-qubit depth
is **constant in the system size**, against the linear growth of a term-by-term
Trotterization. The plumbing that gets an encoding of your own into the transpiler ---
:class:`.CustomF2QLayout` and :class:`.MapperFnEvolutionSynthesis` --- is the part that
carries over to any model.

.. seealso::
   The :ref:`transpilation guide <transpilation_explanation>` for the transpiler stages
   referenced throughout, and the :ref:`mappers guide <mappers_explanation>` for the
   general recipe for writing a custom mapper.

Transfer operators and flow sets
--------------------------------

A :class:`.TransferVertexOperator` is built from vertex operators :math:`V_j` and
transfer operators :math:`T_{jk}`. In terms of fermionic creation and annihilation
operators,

.. math::

   V_j = 1 - 2 a^\dagger_j a_j \, , \qquad
   T_{jk} = -\tfrac{1}{2} \left( a^\dagger_j - a_j \right)
                          \left( a^\dagger_k + a_k \right) \, ,

so :math:`V_j` measures the occupation of mode :math:`j` -- it is :math:`+1` when empty
and :math:`-1` when filled -- while :math:`T_{jk}` transfers a fermion along the
*directed* edge :math:`j \to k`. Note that the two indices of :math:`T_{jk}` are *not*
interchangeable: the **first** index carries the minus combination and the second the
plus, which is what makes :math:`T_{jk}` distinct from :math:`T_{kj}` and gives the edge
its orientation.

These satisfy *mixed* commutation relations: two transfer operators meeting at a shared
site **commute** if the arrows *flow* through that site (one arrives, one leaves) and
**anticommute** if they *clash* (both arrive, or both leave).

The rule is easiest to see on a picture. Drawing the vertices as nodes and each
:math:`T_{jk}` as an arrow from :math:`j` to :math:`k` gives the *directed* interaction
graph -- here for the four-site chain used throughout this guide, which carries both
orientations of every bond:

.. plot::
   :context: close-figs
   :alt: A four-site chain with both orientations of every bond drawn as arrows.

   >>> import rustworkx as rx
   >>> from rustworkx.visualization import mpl_draw
   >>> N = 4
   >>> graph = rx.PyDiGraph()
   >>> _ = graph.add_nodes_from(range(N))
   >>> _ = graph.add_edges_from([(i, i + 1, i) for i in range(N - 1)])
   >>> _ = graph.add_edges_from([(i, i - 1, -i) for i in range(1, N)])
   >>> mpl_draw(
   ...     graph,
   ...     pos={i: (i, -0.1 * i) for i in range(N)},
   ...     labels=lambda v: f"$V_{v}$",
   ...     edge_labels=lambda e: (
   ...         f"$T_{{{e},{e + 1}}}$" if e >= 0 else f"$T_{{{-e},{-e - 1}}}$"
   ...     ),
   ...     with_labels=True,
   ...     node_color="orange",
   ... )
   <Figure size ... with 1 Axes>

Trace the rule on the figure. Take :math:`T_{1,2}` and :math:`T_{2,3}`: one arrow
*arrives* at site 2 and the other *leaves* it, so the fermion flows straight through and
the two operators commute. Take instead :math:`T_{1,2}` and :math:`T_{3,2}`: both arrows
*point into* site 2 -- they clash, and those two anticommute.

This is what makes flow sets possible. Pick a set of arrows forming a directed path and
every pair either meets head-to-tail or does not meet at all, so the whole set commutes
even though the individual operators overlap on shared sites. Evolving under a single flow
set therefore incurs **no Trotter error**, and as :ref:`step 7 <flowsets_synthesis>`
shows, it can also be done at constant circuit depth.

1. The Fermi-Hubbard Hamiltonian
--------------------------------

The 1D Fermi-Hubbard model on :math:`L` sites (spinless, open boundaries) is

.. math::

   H = -t \sum_{j} \left( a^\dagger_j a_{j+1} + a^\dagger_{j+1} a_j \right)
       + U \sum_{j} n_j n_{j+1} \, .

Following Eq. (9) of Ref. [1]_, the hopping term is :math:`t \sum_j (T_{j,j+1} +
T_{j+1,j})`. The interaction is diagonal, so it is expressed through vertex operators
using :math:`n_j = (1 - V_j)/2`:

.. math::

   n_j n_{j+1} = \tfrac{1}{4}\left( 1 - V_j - V_{j+1} + V_j V_{j+1} \right) \, .

Both pieces are built directly as a :class:`.TransferVertexOperator`. Note that
:math:`V_j` is stored as the diagonal entry ``(j, j)``:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from collections import defaultdict
   >>> from qiskit_fermions.operators import TransferVertexOperator
   >>>
   >>> def fermi_hubbard_1d(num_sites, tunneling, interaction):
   ...     data: defaultdict[tuple[tuple[int, int], ...], complex] = defaultdict(complex)
   ...     for j in range(num_sites - 1):
   ...         k = j + 1
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
   >>> num_sites = 4
   >>> hamiltonian = fermi_hubbard_1d(num_sites, tunneling=1.0, interaction=2.0)
   >>> print(format(hamiltonian))
     1.500000e0 +0.000000e0j * ()
   -5.000000e-1 +0.000000e0j * (V(0))
    5.000000e-1 +0.000000e0j * (V(0) V(1))
     1.000000e0 +0.000000e0j * (T(0,1))
     1.000000e0 +0.000000e0j * (T(1,0))
    -1.000000e0 +0.000000e0j * (V(1))
    5.000000e-1 +0.000000e0j * (V(1) V(2))
     1.000000e0 +0.000000e0j * (T(1,2))
     1.000000e0 +0.000000e0j * (T(2,1))
    -1.000000e0 +0.000000e0j * (V(2))
    5.000000e-1 +0.000000e0j * (V(2) V(3))
     1.000000e0 +0.000000e0j * (T(2,3))
     1.000000e0 +0.000000e0j * (T(3,2))
   -5.000000e-1 +0.000000e0j * (V(3))

The interior sites pick up a coefficient of :math:`-1` on :math:`V_j` because they
appear in two bonds, whereas the boundary sites only appear in one. Every term listed here
appears in the graph drawn above: the ``T(j,k)`` terms are its arrows, and the ``V(j)``
terms its nodes.

2. Partitioning into flow sets
------------------------------

Next, label each term with the flow set it belongs to. The
:attr:`~.TransferVertexOperator.groups` attribute exists exactly for this purpose: an
:class:`.Evolution` gate over a grouped operator decomposes into one :class:`.Evolution`
per group (:ref:`step 6 <flowsets_transpile>` puts this to work), so the grouping decided
here at the *fermionic* level survives into the circuit.

For the 1D chain the two hopping flow sets are the east-oriented arrows
(:math:`T_{j,j+1}`) and the west-oriented arrows (:math:`T_{j+1,j}`). The diagonal
interaction terms commute with everything diagonal and form a third group:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> def flow_set_groups(operator):
   ...     groups = []
   ...     for terms, _ in operator.iter_terms():
   ...         match terms:
   ...             case [(left, right)] if right == left + 1:
   ...                 groups.append(0)  # east-oriented: T_{j,j+1}
   ...             case [(left, right)] if right == left - 1:
   ...                 groups.append(1)  # west-oriented: T_{j+1,j}
   ...             case _:
   ...                 groups.append(2)  # diagonal interaction terms
   ...     return groups
   >>>
   >>> hamiltonian.groups = flow_set_groups(hamiltonian)
   >>> hamiltonian.num_groups()
   3
   >>> for index, group in enumerate(hamiltonian.split_out_groups()):
   ...     print(index, sorted(terms for terms, _ in group.iter_terms()))
   0 [[(0, 1)], [(1, 2)], [(2, 3)]]
   1 [[(1, 0)], [(2, 1)], [(3, 2)]]
   2 [[], [(0, 0)], [(0, 0), (1, 1)], [(1, 1)], [(1, 1), (2, 2)], [(2, 2)], [(2, 2), (3, 3)], [(3, 3)]]

Groups 0 and 1 are the two flow sets: each is a directed path along the chain, so by the
flow property all terms within a group commute.

3. Writing the custom encoding
------------------------------

Now the substance of this guide. Section IV B of Ref. [1]_ classifies the forms that a
local fermion-to-qubit encoding can take when restricted to a flow set. Plain
Jordan-Wigner uses :math:`N_q = N_f` qubits and maps every hopping term to a weight-2
Pauli. The classification shows that **spending extra qubits buys structure** in the
encoded operators.

The encoding implemented here is the parity-delocalized construction of Appendix B 2
(Eqs. B4 and B5), which uses one **ancilla qubit**, so :math:`N_q = N_f + 1`. The
fermionic parity is delocalized across a *pair* of qubits:

.. math::

   V_j = Z_j Z_{j+1} \, , \qquad
   T_{j,j+1} = -\tfrac{1}{2} X_{j+1} \, , \qquad
   T_{j+1,j} = +\tfrac{1}{2} Z_j X_{j+1} Z_{j+2} \, .

The payoff is visible immediately: one whole flow set is mapped to **weight-1** Paulis.

.. note::
   The :math:`\tfrac{1}{2}` prefactors are fixed by the normalization :math:`T_{jk}^2 =
   \tfrac{1}{4}`, and the *relative* minus sign between the two orientations is fixed by
   the product identity :math:`T_{j,j+1} = -V_j V_{j+1} T_{j+1,j}`, which the Paulis above
   satisfy. Getting either wrong yields an operator that still obeys the commutation
   relations but no longer represents the same Hamiltonian, so it is worth verifying the
   encoding numerically --- which is exactly what :ref:`step 4 <flowsets_verify>` does.

Writing the encoding means writing a function that maps a **single** generalized transfer
operator to a Pauli string; :func:`.map_transfer_vertex_generators` handles the iteration
over terms and their composition:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit.quantum_info import SparseObservable
   >>> from qiskit_fermions.mappers import map_transfer_vertex_generators
   >>>
   >>> def flow_set_action(action, num_qubits):
   ...     match action:
   ...         case (j, k) if j == k:
   ...             # vertex operator: V_j = Z_j Z_{j+1}
   ...             return SparseObservable.from_sparse_list(
   ...                 [("ZZ", [j, j + 1], 1.0)], num_qubits=num_qubits
   ...             )
   ...         case (j, k) if k == j + 1:
   ...             # east-oriented transfer operator: weight 1
   ...             return SparseObservable.from_sparse_list(
   ...                 [("X", [k], -0.5)], num_qubits=num_qubits
   ...             )
   ...         case (j, k) if k == j - 1:
   ...             # west-oriented transfer operator: weight 3
   ...             return SparseObservable.from_sparse_list(
   ...                 [("ZXZ", [k, k + 1, k + 2], 0.5)], num_qubits=num_qubits
   ...             )
   ...         case _:
   ...             raise ValueError(f"not a nearest-neighbour transfer operator: {action}")
   >>>
   >>> def flow_set_encoding(operator, num_qubits):
   ...     return map_transfer_vertex_generators(
   ...         operator,
   ...         lambda action: flow_set_action(action, num_qubits),
   ...         identity=lambda: SparseObservable.identity(num_qubits),
   ...         compose=SparseObservable.compose,
   ...     ).simplify()

.. important::
   The signature ``(operator, num_qubits)`` is what :attr:`.MapperFnEvolutionSynthesis.mapper_fn`
   expects, and the return type must be a
   :class:`~qiskit.quantum_info.SparseObservable`. Keeping to that contract is what makes
   the function directly usable as a transpiler plugin in :ref:`step 6 <flowsets_transpile>`.

Applying it to the Hamiltonian from step 1 gives the encoded operator on
:math:`N_f + 1 = 5` qubits:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> num_qubits = num_sites + 1
   >>> encoded = flow_set_encoding(hamiltonian, num_qubits)
   >>> for label, indices, coeff in sorted(encoded.to_sparse_list()):
   ...     print(f"{label:3s} {str(indices):12s} {coeff.real:+.2f}")
       []           +1.50
   X   [1]          -0.50
   X   [2]          -0.50
   X   [3]          -0.50
   ZXZ [0, 1, 2]    +0.50
   ZXZ [1, 2, 3]    +0.50
   ZXZ [2, 3, 4]    +0.50
   ZZ  [0, 1]       -0.50
   ZZ  [0, 2]       +0.50
   ZZ  [1, 2]       -1.00
   ZZ  [1, 3]       +0.50
   ZZ  [2, 3]       -1.00
   ZZ  [2, 4]       +0.50
   ZZ  [3, 4]       -0.50

Fourteen terms, in exactly three shapes: the hopping has become ``X`` and ``ZXZ``, the
interaction ``ZZ``, and the leading ``[]`` is the identity contributing only a global
phase. Note that the ``ZZ`` terms reach both nearest *and* next-nearest neighbours ---
:math:`V_jV_{j+1}` spans qubits :math:`j` through :math:`j+2` --- which is the cost side of
delocalizing the parity.

Grouping by flow set exposes the asymmetry the construction buys. Listing the Pauli
weights present in each group:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> for index, group in enumerate(hamiltonian.split_out_groups()):
   ...     weights = {len(indices) for _, indices, _ in flow_set_encoding(group, num_qubits).to_sparse_list()}
   ...     print(index, sorted(weights))
   0 [1]
   1 [3]
   2 [0, 2]

The east-oriented flow set (group 0) consists **entirely of weight-1 Paulis**. Its time
evolution is therefore a layer of single-qubit rotations and needs *no entangling gates at
all* -- whereas under Jordan-Wigner every hopping term is weight-2 and requires them. This
is the space-time trade-off of Ref. [1]_ in its simplest form: one extra qubit converts
half of the hopping Hamiltonian into single-qubit rotations.

The west-oriented flow set (group 1) pays for it at weight 3, and the interaction (group 2)
sits at weight 2, with the weight-0 entry being the identity. Nothing here is yet a
*circuit* claim --- what a synthesis makes of these weights is the subject of steps 5
and 6.

.. _flowsets_verify:

4. Verifying the encoding
-------------------------

A custom encoding is only useful if it is *faithful*, and a hand-written one deserves to
be checked rather than trusted. Because this encoding uses :math:`N_f + 1` qubits, the
qubit Hilbert space is twice as large as the fermionic one, so the encoded Hamiltonian
cannot equal the Jordan-Wigner one term by term. The two are related by an isometry
instead.

**Where the isometry comes from.** Under Jordan-Wigner a basis state simply *is* the
occupation string: qubit :math:`j` holds :math:`n_j`. This encoding instead stores
*cumulative parities*. Define

.. math::

   b_0 = 0 \, , \qquad b_{j+1} = n_0 \oplus n_1 \oplus \dots \oplus n_j \, ,

so bit :math:`b_{j+1}` records the parity of all occupations up to and including site
:math:`j`. There are :math:`N_f + 1` such bits for :math:`N_f` occupations, which is
where the ancilla comes from. The occupation is recovered as the *difference* between
neighbouring bits, :math:`n_j = b_j \oplus b_{j+1}` --- so an occupied site shows up as a
**domain wall** between two adjacent bits, which is why this is called a domain-wall (or
Kramers-Wannier) encoding. That is also exactly why :math:`V_j = Z_j Z_{j+1}`: the
product of two neighbouring :math:`Z`\ s reads off precisely that difference.

Writing this map out for a few states makes the structure concrete (note that
:math:`b_0 = 0` is fixed, so only *half* of the :math:`2^{N_f+1}` qubit states are in the
image --- hence an isometry rather than a unitary):

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> def domain_wall_bits(occupations):
   ...     parity = 0
   ...     bits = [0]
   ...     for occupation in occupations:
   ...         parity ^= occupation
   ...         bits.append(parity)
   ...     return bits
   >>>
   >>> for occupations in ([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]):
   ...     bits = domain_wall_bits(occupations)
   ...     # little-endian: reversed() prints qubit 0 on the right, as Qiskit labels do
   ...     print("".join(map(str, occupations)), "->", "".join(str(b) for b in reversed(bits)))
   0000 -> 00000
   1000 -> 11110
   1100 -> 00010
   1010 -> 00110

A single fermion on site 0 flips *every* cumulative parity above it, which is the
delocalization at work. Collecting these columns into a matrix gives the isometry
:math:`W`, and the encoding is faithful precisely when it *intertwines* the two
Hamiltonians:

.. math::

   \hat{H}_{\text{flow set}} \, W = W \, \hat{H}_{\text{JW}} \, .

In words: mapping a fermionic state into the qubit space and then evolving it there must
give the same result as evolving it in the fermionic space first and mapping afterwards.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import numpy as np
   >>> from qiskit.quantum_info import SparsePauliOp
   >>> from qiskit_fermions.mappers.library import jordan_wigner, transfer_vertex_to_fermion
   >>>
   >>> def to_matrix(observable):
   ...     # SparseObservable carries no dense-matrix method, so route through
   ...     # SparsePauliOp. This is only ever needed for the checks below --
   ...     # the encoding itself never leaves SparseObservable.
   ...     return SparsePauliOp.from_sparse_observable(observable).to_matrix()
   >>>
   >>> def domain_wall_isometry(num_sites):
   ...     isometry = np.zeros((2 ** (num_sites + 1), 2**num_sites), dtype=complex)
   ...     for index in range(2**num_sites):
   ...         occupations = [(index >> j) & 1 for j in range(num_sites)]
   ...         bits = domain_wall_bits(occupations)
   ...         isometry[sum(b << j for j, b in enumerate(bits)), index] = 1.0
   ...     return isometry
   >>>
   >>> isometry = domain_wall_isometry(num_sites)
   >>> np.allclose(isometry.conj().T @ isometry, np.eye(2**num_sites))
   True
   >>>
   >>> jw_matrix = to_matrix(
   ...     jordan_wigner(transfer_vertex_to_fermion(hamiltonian), num_sites).simplify()
   ... )
   >>> flow_matrix = to_matrix(flow_set_encoding(hamiltonian, num_qubits))
   >>> np.allclose(flow_matrix @ isometry, isometry @ jw_matrix)
   True

The intertwining relation holds exactly. As a corollary the spectra must agree, with
every Jordan-Wigner eigenvalue appearing twice in the larger space --- once for each
value of the unconstrained global parity:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> jw_spectrum = np.linalg.eigvalsh(jw_matrix)
   >>> flow_spectrum = np.linalg.eigvalsh(flow_matrix)
   >>> np.allclose(
   ...     np.sort(flow_spectrum),
   ...     np.sort(np.concatenate([jw_spectrum, jw_spectrum])),
   ... )
   True
   >>> print(f"{jw_spectrum[0]:.10f}  {flow_spectrum[0]:.10f}")
   -1.8557725066  -1.8557725066

5. Building the fermionic circuit
---------------------------------

Before any qubits enter the picture, the dynamics are expressed on the *fermionic* level.
A :class:`.FermionicCircuit` acts on fermionic modes rather than qubits, and an
:class:`.Evolution` gate over the Hamiltonian is the whole circuit:

.. plot::
   :context: close-figs
   :include-source:
   :alt: A single Evolution gate spanning all four fermionic modes.

   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import Evolution
   >>>
   >>> total_time = 1.0
   >>> circuit = FermionicCircuit(num_sites)
   >>> circuit.append(Evolution(num_sites, hamiltonian, time=total_time), circuit.modes)
   >>> circuit.draw("mpl")
   <Figure size ... with 1 Axes>

One opaque box over all four modes. The interesting part is what it decomposes into:
:class:`.Evolution` splits itself group-by-group whenever its operator has
:attr:`~qiskit_fermions.operators.TransferVertexOperator.groups` assigned. So a single
:meth:`~qiskit.circuit.QuantumCircuit.decompose` turns it into **one** :class:`.Evolution`
**per flow set**, ordered by group index --- east, west, then the diagonal interaction,
carrying 3, 3 and 8 terms respectively, exactly the groups from step 2:

.. plot::
   :context: close-figs
   :include-source:
   :alt: The same circuit decomposed into three Evolution gates, one per flow set.

   >>> flow_set_circuit = circuit.decompose()
   >>> flow_set_circuit.draw("mpl")
   <Figure size ... with 1 Axes>

This is the step that makes the grouping matter. Each of the three gates is mapped and
synthesized independently, so the partitioning decided at the fermionic level in step 2 is
what the transpiler ends up acting on. Skip the decomposition and the grouping is simply
ignored.

.. _flowsets_transpile:

6. Transpiling with the custom encoding
---------------------------------------

The encoding is now ready to be used by the transpiler. Two more pieces are needed:

**A layout.** The default :class:`.TrivialF2QLayout` assumes one qubit per mode, which is
wrong here. :class:`.CustomF2QLayout` associates the fermionic register with a
:class:`~qiskit.circuit.QuantumRegister` of a different size --- this is how ancilla
qubits enter the pipeline.

**A synthesis plugin.** :class:`.MapperFnEvolutionSynthesis` takes the mapper function
directly, maps the :class:`.Evolution` gate's Hamiltonian with it, and emits a
:class:`~qiskit.circuit.library.PauliEvolutionGate` preserving the :math:`e^{-itH}`
convention. Its :attr:`~.MapperFnEvolutionSynthesis.product_formula` is left at its default
for now; step 7 is where a custom one gets slotted in.

Since every comparison below re-runs this same pipeline, it is worth wrapping once:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit.circuit import QuantumRegister
   >>> from qiskit.passmanager import MultiStagePassManager
   >>> from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
   >>> from qiskit_fermions.transpiler.passes import (
   ...     CustomF2QLayout,
   ...     F2QSynthesis,
   ...     MapperFnEvolutionSynthesis,
   ... )
   >>>
   >>> def flow_set_pass_manager(mode_register, product_formula=None):
   ...     '''Build the pipeline mapping `mode_register` with the flow-set encoding.'''
   ...     synthesis = F2QSynthesis()
   ...     synthesis.methods["Evolution"] = MapperFnEvolutionSynthesis(
   ...         flow_set_encoding, product_formula=product_formula
   ...     )
   ...     # this encoding spends one ancilla on top of the fermionic modes
   ...     qubit_register = QuantumRegister(mode_register.size + 1, "q")
   ...     return MultiStagePassManager(
   ...         input=FermionicCircuitToDAG(),
   ...         layout=CustomF2QLayout({mode_register: qubit_register}),
   ...         synthesis=synthesis,
   ...         output=QuantumDAGToCircuit(),
   ...     )
   >>>
   >>> pass_manager = flow_set_pass_manager(flow_set_circuit.register)
   >>> qubit_circuit = pass_manager.run(flow_set_circuit)
   >>> qubit_circuit.num_qubits
   5

Four fermionic modes have become a five-qubit circuit, with the ancilla supplied by the
custom layout.

Now look at what the resulting circuit actually contains. Decomposing repeatedly reduces
the evolution gates all the way down to ``U`` and ``CX``, so that the two-qubit cost of the
two synthesis strategies compared below is measured in the same currency:
:meth:`~qiskit.circuit.QuantumCircuit.count_ops` then reports the ``CX`` count, and
:meth:`~qiskit.circuit.QuantumCircuit.depth` takes a filter function to measure the
two-qubit depth:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> decomposed = qubit_circuit.decompose(reps=6)
   >>> decomposed.count_ops()["cx"]
   26
   >>> decomposed.depth(lambda instruction: len(instruction.qubits) == 2)
   23

Twenty-six ``CX`` gates in twenty-three layers, so almost nothing runs in parallel. Drawing
the circuit shows why:

.. plot::
   :context: close-figs
   :include-source:
   :alt: The transpiled circuit under LieTrotter's term-by-term synthesis, a long serial chain of CX ladders.

   >>> decomposed.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

Every term has been compiled in isolation: a basis change, a ``CX`` ladder down to a single
rotation, then the ladder undone. The weight-1 flow set really is free --- its terms are
lone rotations with no ladder around them --- but each weight-3 term pays for four
``CX``\ s, and because the ladders are emitted one term after another nothing overlaps.

So the grouping *has* been honoured at the level of the pipeline --- three flow sets, three
mapped :class:`~qiskit.circuit.library.PauliEvolutionGate`\ s --- but it buys almost
nothing, because :class:`~qiskit.synthesis.LieTrotter` synthesizes each one **term by
term**. It does not know that the terms handed to it mutually commute, so it still compiles
them one after another. The structural advantage is present in the *operator* and preserved by
the *decomposition*, then discarded by the *synthesis*. That last step is what the next
section replaces.

.. _flowsets_synthesis:

7. A flow-set-aware synthesis
-----------------------------

Recovering the advantage requires telling the synthesis about the structure. This is
Section IV C of Ref. [1]_: because all terms of a flow set commute, they can be
**simultaneously diagonalized** by a single Clifford circuit, rotated in one shared basis,
and rotated back --- instead of once per term.

For this encoding the Clifford is remarkably simple. Conjugating by a chain of ``CZ``
gates on nearest neighbours maps

.. math::

   X_{j+1} \; \longmapsto \; Z_j X_{j+1} Z_{j+2}

simultaneously for **every** :math:`j`. So one ``CZ`` chain turns the entire weight-3
(west) flow set into a layer of independent single-qubit :math:`X` rotations. All the
:class:`~qiskit.synthesis.EvolutionSynthesis` interface asks for is a
:meth:`~qiskit.synthesis.EvolutionSynthesis.synthesize` method; ``reps`` is here because a
product formula is also where the number of Trotter steps belongs, exactly as in Qiskit's
own :class:`~qiskit.synthesis.LieTrotter`:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit.circuit import QuantumCircuit
   >>> from qiskit.synthesis.evolution import EvolutionSynthesis
   >>>
   >>> def cz_brickwork(circuit):
   ...     # emit even and odd bonds as two layers, so the CZs are scheduled in
   ...     # parallel: a chain of overlapping CZs has depth 2, not num_qubits - 1
   ...     for offset in (0, 1):
   ...         for j in range(offset, circuit.num_qubits - 1, 2):
   ...             circuit.cz(j, j + 1)
   >>>
   >>> # The interaction reaches nearest (distance 1) and next-nearest (distance 2)
   >>> # neighbours, so every interior qubit carries four Rzz gates and can only do one at
   >>> # a time: four layers is the best any schedule can do. These offsets achieve it, as
   >>> # (distance, modulus, starting offsets) selecting the gates of one layer.
   >>> RZZ_SCHEDULE = ((1, 2, (0,)), (2, 4, (0, 1)), (1, 2, (1,)), (2, 4, (2, 3)))
   >>>
   >>> def rzz_layers(circuit, angles):
   ...     # emit layer by layer, since Qiskit's depth follows the order gates were added
   ...     for distance, modulus, offsets in RZZ_SCHEDULE:
   ...         for (j, k), angle in angles.items():
   ...             if k - j == distance and j % modulus in offsets:
   ...                 circuit.rzz(angle, j, k)
   >>>
   >>> class FlowSetSynthesis(EvolutionSynthesis):
   ...     '''Synthesize one flow set of the encoded Hamiltonian at a time.'''
   ...
   ...     preserve_order = True
   ...
   ...     def __init__(self, reps=1):
   ...         self.reps = reps
   ...
   ...     def synthesize(self, evolution):
   ...         circuit = QuantumCircuit(evolution.operator.num_qubits)
   ...         east, west, diagonal = [], [], {}
   ...
   ...         for label, indices, coeff in evolution.operator.to_sparse_list():
   ...             # as in Qiskit's own product formulas, `reps` Trotter steps divide the
   ...             # time -- and hence every rotation angle -- by `reps`
   ...             angle = 2 * evolution.time * coeff.real / self.reps
   ...             match label, indices:
   ...                 case "", _:  # the identity contributes only a global phase
   ...                     circuit.global_phase -= self.reps * angle / 2
   ...                 case "X", [qubit]:  # east flow set: already weight-1
   ...                     east.append((qubit, angle))
   ...                 case "ZXZ", [j, k, l] if (k, l) == (j + 1, j + 2):
   ...                     west.append((k, angle))  # needs the CZ conjugation below
   ...                 case "ZZ", [j, k]:  # interaction: diagonal, see rzz_layers above
   ...                     diagonal[min(j, k), max(j, k)] = angle
   ...                 case _:
   ...                     raise ValueError(f"unexpected Pauli term: {label} on {indices}")
   ...
   ...         for _ in range(self.reps):
   ...             # east flow set: bare rotations, no entangling gates at all
   ...             for qubit, angle in east:
   ...                 circuit.rx(angle, qubit)
   ...
   ...             # west flow set: one CZ chain diagonalizes the WHOLE set at once
   ...             if west:
   ...                 cz_brickwork(circuit)
   ...                 for qubit, angle in west:
   ...                     circuit.rx(angle, qubit)
   ...                 cz_brickwork(circuit)
   ...
   ...             rzz_layers(circuit, diagonal)
   ...
   ...         return circuit

.. note::
   Two details are easy to miss. A custom
   :class:`~qiskit.synthesis.EvolutionSynthesis` must expose a ``preserve_order``
   attribute, which the high-level-synthesis pass reads.

   And both helpers emit their gates in explicit layers rather than in term order, because
   :meth:`~qiskit.circuit.QuantumCircuit.depth` schedules gates *as soon as possible in the
   order they were added*. Since all the gates within one of these sets commute, that order
   is ours to choose, and choosing it badly is what a term-by-term synthesis does: emitting
   the overlapping ``CZ`` chain sequentially reports a depth growing with the qubit count
   rather than the constant 2 the hardware could achieve.

Selecting it requires nothing more than handing it to
:attr:`.MapperFnEvolutionSynthesis.product_formula` --- which is the argument
``flow_set_pass_manager`` takes. Nothing else about the pipeline, or the fermionic circuit
it runs on, changes:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> pass_manager = flow_set_pass_manager(flow_set_circuit.register, FlowSetSynthesis())
   >>>
   >>> flow_circuit = pass_manager.run(flow_set_circuit).decompose(reps=6)
   >>> flow_circuit.count_ops()["cx"]
   22
   >>> flow_circuit.depth(lambda instruction: len(instruction.qubits) == 2)
   12

.. plot::
   :context: close-figs
   :include-source:
   :alt: The flow-set-aware circuit, markedly shorter than the previous one, with entangling gates stacked into parallel layers.

   >>> flow_circuit.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

From depth 23 with 26 ``CX`` gates down to depth 12 with 22: fewer entangling gates *and*
nearly half the depth. Comparing the two pictures, the difference is not that the gates are
cheaper but that they *stack*. The serial ladders are gone; entangling gates now sit above
one another in shared layers, because the whole west flow set is rotated in a single shared
basis and the interaction is emitted in layers of disjoint pairs.

.. note::
   Both drawings show the circuit after ``decompose(reps=6)``, which rewrites everything
   into the ``U`` and ``CX`` basis --- so the ``CZ``, ``Rx`` and ``Rzz`` gates emitted by
   ``FlowSetSynthesis`` are not visible as such. That is deliberate: it puts both circuits
   in the same basis, which is the only way the two-qubit counts and depths above are
   comparable. It is the *layer structure* that is worth reading off these figures, not the
   gate labels.

8. Constant depth at scale
--------------------------

The interesting question is not the count at four modes but how it *scales*. The two
``CZ`` brickwork layers do not grow with the chain length, so the two-qubit depth of one
Trotter step should be constant. Measuring it directly on the full Fermi-Hubbard
Hamiltonian, well past what could be simulated by brute force:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> def evolution_stats(num_sites, product_formula):
   ...     operator = fermi_hubbard_1d(num_sites, tunneling=1.0, interaction=2.0)
   ...     operator.groups = flow_set_groups(operator)
   ...
   ...     circuit = FermionicCircuit(num_sites)
   ...     circuit.append(Evolution(num_sites, operator, time=total_time), circuit.modes)
   ...     circuit = circuit.decompose()  # split into one Evolution per flow set
   ...
   ...     pass_manager = flow_set_pass_manager(circuit.register, product_formula)
   ...     decomposed = pass_manager.run(circuit).decompose(reps=6)
   ...     two_qubit = lambda instruction: len(instruction.qubits) == 2
   ...     return decomposed.depth(two_qubit), decomposed.count_ops()["cx"]
   >>>
   >>> from qiskit.synthesis import LieTrotter
   >>>
   >>> sites = [4, 10, 20, 50, 100]
   >>> flow = [evolution_stats(n, FlowSetSynthesis()) for n in sites]
   >>> lie_trotter = [evolution_stats(n, LieTrotter()) for n in sites]
   >>>
   >>> [depth for depth, _ in flow]
   [12, 12, 12, 12, 12]
   >>> [depth for depth, _ in lie_trotter]
   [23, 47, 87, 207, 407]

The two-qubit **depth stays at 12** from 4 modes to 100, while :class:`~qiskit.synthesis.LieTrotter`
grows linearly to 407 --- the flow sets let it parallelize a little *within* the pipeline, but
each set is still Trotterized term by term, so the growth is only slowed, not removed:

.. plot::
   :context: close-figs
   :include-source:
   :alt: Two-qubit depth against number of modes; LieTrotter grows linearly while the flow-set synthesis stays flat at 12.

   >>> import matplotlib.pyplot as plt
   >>>
   >>> figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
   >>> for panel, index, title in zip(axes, (0, 1), ("two-qubit depth", "CX count")):
   ...     _ = panel.plot(sites, [stat[index] for stat in lie_trotter], "o-", label="Lie")
   ...     _ = panel.plot(sites, [stat[index] for stat in flow], "s-", label="flow set")
   ...     _ = panel.set(xlabel="modes", title=title)
   ...     _ = panel.legend()
   >>> figure
   <Figure size ... with 2 Axes>

.. plot::
   :context:
   :nofigs:

   Release the figure above now that the plot directive has captured it. Sybil shares
   pyplot's state across all guides, so a figure left open here gets drawn into by the
   next guide that calls ``mpl_draw`` without an explicit ``ax=``. The directive's
   ``close-figs`` option does not cover that, since Sybil only executes the doctests.

   >>> plt.close(figure)

The gate *count* grows linearly under both --- there are :math:`O(N)` terms to apply, and
that is unavoidable --- but the flow-set synthesis applies them in a constant number of
parallel layers. This is the constant-depth result of Ref. [1]_.

.. note::
   The depth 12 splits across the three flow sets as 0 + 4 + 8, at every chain length. The
   east flow set contributes **zero**: it is bare ``Rx`` rotations. The west flow set costs 4,
   from the two ``CZ`` brickwork layers going in and the two coming out --- dropping the
   interaction leaves exactly that, a two-qubit depth of 4 at any chain length. The remaining
   8 is the interaction, whose ``Rzz`` terms act on both nearest and next-nearest neighbours
   and so need four disjoint layers, each costing two ``CX``\ s. Four is optimal here: every
   interior qubit carries four ``Rzz`` gates --- two to each neighbour and two to each
   next-nearest one --- and can only take part in one at a time, so no schedule can do
   better.

   That two thirds of the depth is the interaction shows where this encoding really excels:
   at the hopping, and so at the dynamics of a free-fermion chain. Do note, though, that the
   interaction does not commute with the hopping --- leaving it out is not an approximation to
   Fermi-Hubbard but a different model. Where the balance falls for a given problem is worth
   exploring.

9. Checking the dynamics
------------------------

Finally, the physics. Under the domain-wall map an occupation pattern becomes a
cumulative-parity string, so the initial state is prepared differently than under
Jordan-Wigner --- a practical consequence of delocalizing the parity that is easy to
overlook. The helper from step 4 gives the right label:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> occupations = [1, 0, 1, 0]
   >>> initial_label = "".join(str(b) for b in reversed(domain_wall_bits(occupations)))
   >>> initial_label
   '00110'

Now compare ``FlowSetSynthesis`` against :class:`~qiskit.synthesis.LieTrotter` at increasing
numbers of Trotter steps, measuring the fidelity of the evolved state against exact matrix
exponentiation. The step count is a property of the *product formula*, so both columns run
the very same fermionic circuit --- ``circuit`` from step 5, a single :class:`.Evolution`
gate at the full time --- and let the synthesis subdivide it:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> from qiskit.quantum_info import Statevector
   >>> from scipy.linalg import expm
   >>>
   >>> reference = expm(-1j * total_time * flow_matrix) @ Statevector.from_label(initial_label).data
   >>>
   >>> def trotter_fidelity(product_formula):
   ...     # `circuit` is the undecomposed one from step 5 -- see the note below
   ...     pass_manager = flow_set_pass_manager(circuit.register, product_formula)
   ...     # NOTE: decompose() is essential -- see the warning below
   ...     decomposed = pass_manager.run(circuit).decompose(reps=6)
   ...     evolved = Statevector.from_label(initial_label).evolve(decomposed)
   ...     fidelity = abs(np.vdot(evolved.data, reference)) ** 2
   ...     two_qubit = lambda instruction: len(instruction.qubits) == 2
   ...     return fidelity, decomposed.count_ops()["cx"], decomposed.depth(two_qubit)
   >>>
   >>> for steps in (1, 2, 4, 8):
   ...     flow_set = trotter_fidelity(FlowSetSynthesis(reps=steps))
   ...     trotter = trotter_fidelity(LieTrotter(reps=steps))
   ...     print(
   ...         f"{steps} step(s):  flow set {flow_set[0]:.6f}"
   ...         f" ({flow_set[1]:3d} CX, depth {flow_set[2]:3d})"
   ...         f"   Lie {trotter[0]:.6f} ({trotter[1]:3d} CX, depth {trotter[2]:3d})"
   ...     )
   1 step(s):  flow set 0.590661 ( 22 CX, depth  12)   Lie 0.148587 ( 26 CX, depth  26)
   2 step(s):  flow set 0.926656 ( 44 CX, depth  24)   Lie 0.212137 ( 52 CX, depth  48)
   4 step(s):  flow set 0.982053 ( 88 CX, depth  48)   Lie 0.723415 (104 CX, depth  92)
   8 step(s):  flow set 0.995395 (176 CX, depth  96)   Lie 0.925807 (208 CX, depth 180)

The flow-set synthesis wins on **all three** axes at once: higher fidelity, fewer entangling
gates *and* roughly half the two-qubit depth at every step count. This is not a
depth-versus-accuracy trade --- the Trotter error is genuinely smaller because each flow set
is evolved *exactly*, so the only error left comes from splitting the three groups against
each other, rather than from splitting all fourteen terms.

.. note::
   Which circuit goes in matters: the *undecomposed* one from step 5, holding a single
   :class:`.Evolution` gate over all fourteen terms --- not the group-wise
   :meth:`~qiskit.circuit.QuantumCircuit.decompose` of :ref:`step 6 <flowsets_transpile>`.
   That is what keeps the default column an honest baseline. Handing it the decomposed
   circuit would let it inherit the flow-set partitioning for free, and since Trotterizing a
   set of *commuting* terms is exact, its fidelities would become identical to the flow-set
   column's.

   The flip side is that ``FlowSetSynthesis`` has to recover the flow sets itself, from the
   Pauli labels it is handed. That works here because the encoding gives each set a
   recognizable shape, but it is the less general route --- the group-wise
   :meth:`~qiskit.circuit.QuantumCircuit.decompose` works for *any* grouping, not just one a
   synthesis plugin happens to be able to reverse-engineer.

.. warning::
   The :meth:`~qiskit.circuit.QuantumCircuit.decompose` call above is essential.
   :meth:`~qiskit.quantum_info.Statevector.evolve` and
   :class:`~qiskit.quantum_info.Operator` use a
   :class:`~qiskit.circuit.library.PauliEvolutionGate`'s *exact* definition and bypass a
   custom synthesis entirely. Verifying against an un-decomposed circuit reports perfect
   fidelity even for a deliberately wrong synthesis.

As a final cross-check, the site densities :math:`\langle n_j(t) \rangle = (1 - \langle
V_j \rangle)/2` computed in the encoded space must reproduce the Jordan-Wigner result:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> time = 0.8
   >>> state_jw = np.zeros(2**num_sites, dtype=complex)
   >>> state_jw[sum(o << j for j, o in enumerate(occupations))] = 1.0
   >>> evolved_jw = expm(-1j * time * jw_matrix) @ state_jw
   >>> evolved_flow = expm(-1j * time * flow_matrix) @ (isometry @ state_jw)
   >>>
   >>> for j in range(num_sites):
   ...     v_flow = to_matrix(
   ...         SparseObservable.from_sparse_list([("ZZ", [j, j + 1], 1.0)], num_qubits)
   ...     )
   ...     v_jw = to_matrix(SparseObservable.from_sparse_list([("Z", [j], 1.0)], num_sites))
   ...     density_flow = (1 - np.vdot(evolved_flow, v_flow @ evolved_flow).real) / 2
   ...     density_jw = (1 - np.vdot(evolved_jw, v_jw @ evolved_jw).real) / 2
   ...     print(f"site {j}:  {density_flow:.10f}  {density_jw:.10f}")
   site 0:  0.6507935456  0.6507935456
   site 1:  0.5963201678  0.5963201678
   site 2:  0.3365016619  0.3365016619
   site 3:  0.4163846247  0.4163846247

The densities agree to ten decimal places.

What to take away
-----------------

- **Group before mapping.** Flow sets are defined on the directed fermionic interaction
  graph, so the grouping is chosen while problem structure is still available. The
  :attr:`~.TransferVertexOperator.groups` attribute carries it into the circuit, where a
  single :meth:`~qiskit.circuit.QuantumCircuit.decompose` turns it into one
  :class:`.Evolution` gate per flow set.
- **Ancillas buy structure.** Spending one extra qubit turned an entire flow set into
  weight-1 Paulis, whose evolution needs no entangling gates. Ref. [1]_ observes this
  trade-off generally: larger qubit-to-fermion ratios admit shallower evolution circuits.
- **A custom encoding is one function.** Map a single generalized transfer operator to a
  Pauli string; :func:`.map_transfer_vertex_generators` and
  :class:`.MapperFnEvolutionSynthesis` do the rest.
- **The mapping and the grouping alone are not enough.** Splitting the evolution per flow
  set only slowed the linear growth in depth; a term-by-term synthesis of each set still
  threw away the fact that its terms commute. Only supplying a flow-set-aware
  :class:`~qiskit.synthesis.EvolutionSynthesis` made the depth constant. Custom encodings,
  grouping and custom synthesis are complementary --- all three are needed.
- **Verify it.** An encoding that satisfies the commutation relations may still represent
  a different Hamiltonian if a prefactor or sign is off. Checking an intertwining relation
  (or, more cheaply, the spectrum) catches this.

.. note::
   The ``FlowSetSynthesis`` above is deliberately written for *this* encoding on a 1D
   chain: it pattern-matches the specific Pauli labels the encoding produces and raises on
   anything else, rather than silently falling back to a generic product formula. A general
   implementation would read the flow-set structure from the operator's
   :attr:`~.TransferVertexOperator.groups` and derive the diagonalizing Clifford from the
   stabilizer group, as described in Section IV C of Ref. [1]_.

   Note also that delocalizing the parity makes the interaction term :math:`n_j n_{j+1}`
   weight-2 rather than diagonal in single qubits, which is the counterpart cost to the
   cheaper hopping terms.

.. _flowsets_2d:

Where this goes next: two dimensions
------------------------------------

Everything above is one-dimensional, and deliberately so: on a chain the flow sets are
just the two arrow orientations, and the encoding needs a single ancilla. Neither
simplification survives in two dimensions, which is where local encodings earn their
keep.

On a planar lattice, a flow set is a directed path (or a union of vertex-disjoint directed
paths) through the interaction graph, and finding a small set of such paths that covers
every edge becomes a combinatorial problem in its own right rather than a two-line
classification. The encodings also change character: instead of a single ancilla for the
whole chain, they add one per lattice site or cell. Verstraete-Cirac pairs every fermionic
site with its own auxiliary qubit, a qubit-to-mode ratio of :math:`2`, while Derby-Klassen places
one at the centre of each odd plaquette for a ratio of :math:`3/2`. The register layout then
has genuine two-dimensional structure, and the stabilizer group that must be projected onto
is no longer trivial --- the encoded Hamiltonian is only faithful on a *subspace*, which
changes both state preparation and the verification recipe from
:ref:`step 4 <flowsets_verify>`.

The pieces this guide builds are exactly the ones that carry over ---
:class:`.CustomF2QLayout` for a register whose size differs from the mode count,
:func:`.map_transfer_vertex_generators` for the encoding itself,
:class:`.MapperFnEvolutionSynthesis` for wiring it into the transpiler, and a custom
:class:`~qiskit.synthesis.EvolutionSynthesis` for making the flow-set structure pay off.
Only the encoding and the flow-set decomposition are different.

.. [1] A. Gandon, S. Piccinelli, M. Rossmannek, F. Tacchino, A. Baiardi, J. Nys, and
       I. Tavernelli, Stabilizer-based quantum simulation of fermion dynamics with local
       qubit encodings (2026), `arXiv:2512.11418v2 <https://arxiv.org/abs/2512.11418v2>`_.
