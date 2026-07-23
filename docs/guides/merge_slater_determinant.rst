.. _merge_slater_determinant_explanation:

Merge a mode initialization and rotation into a Slater determinant preparation
==============================================================================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available via the C API in a future release.

An :class:`.InitializeModes` gate declares a reference mode occupation; an
:class:`.OrbitalRotation` gate rotates the single-particle basis. Placed back to back, the two
gates *prepare a Slater determinant*: the modes start in the reference determinant and are then
rotated into the target orbitals. The :class:`.PrepareSlaterDeterminant` gate captures exactly that
composition -- and because it knows the reference occupation, transpiling it can use the rectangular
:func:`~qiskit_fermions.linalg.givens_decomposition_slater`, which realizes only the *occupied*
orbitals, instead of the full square decomposition an :class:`.OrbitalRotation` alone requires. That
is a strictly cheaper synthesis (see the payoff at the end of this guide).

The :class:`.MergeSlaterDeterminantPreparation` transpiler pass detects the
:class:`.InitializeModes`-then-:class:`.OrbitalRotation` pattern in a :class:`.FermionicCircuit` and
rewrites it into :class:`.PrepareSlaterDeterminant` gate(s), so a later synthesis stage picks up the
reduced decomposition automatically. Under simulation the rewrite is a no-op on the state:
:class:`.PrepareSlaterDeterminant` is *validate-then-rotate* (it validates the reference occupation
and then applies the rotation, exactly as the two separate gates do), so the merge only unlocks the
cheaper synthesis without changing the prepared state.

Throughout this guide we inspect a circuit by listing its fermionic gates and the modes they act on,
before and after the pass. This small helper does that by running the pass on a copy of the circuit
and walking the resulting :class:`.FermionicDAGCircuit`:

.. doctest::

   >>> from qiskit_fermions.transpiler import FermionicCircuitToDAG
   >>> from qiskit_fermions.transpiler.passes import MergeSlaterDeterminantPreparation
   >>>
   >>> def show_merged(circuit):
   ...     dag = FermionicCircuitToDAG().run(circuit)
   ...     merged = MergeSlaterDeterminantPreparation().run(dag)
   ...     for node in merged.topological_op_nodes():
   ...         modes = sorted(merged.find_bit(qubit).index for qubit in node.qargs)
   ...         print(f"{node.op.name} on modes {modes}")

What gets merged
----------------

The pass recognizes three patterns, all keyed off the block-spin mode convention that
:class:`.InitializeModes` and :class:`.OrbitalRotation` use: for a system with ``norb`` spatial
orbitals the modes ``0..norb`` are the spin-alpha sector and ``norb..2*norb`` are the spin-beta
sector. Slater determinant preparation is done per spin sector because each sector has its own
(electrons, orbitals) shape.

Full-register (or spinless)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest pattern: an :class:`.InitializeModes` immediately followed by an
:class:`.OrbitalRotation` on the *same* modes. This is the spinless case (one determinant over all
modes) and also a spinful circuit whose rotation spans both sectors at once.

.. doctest::

   >>> import numpy as np
   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import InitializeModes, OrbitalRotation
   >>>
   >>> circuit = FermionicCircuit(4)
   >>> circuit.append(InitializeModes([1, 1, 0, 0]), circuit.modes)
   >>> circuit.append(OrbitalRotation(np.eye(4)), circuit.modes)
   >>> show_merged(circuit)
   PrepareSlaterDeterminant on modes [0, 1, 2, 3]

Per spin sector
~~~~~~~~~~~~~~~

The same shape restricted to one spin half. Here two independent initialize-then-rotate pairs, one
per sector, each fuse on their own into a :class:`.PrepareSlaterDeterminant`:

.. doctest::

   >>> circuit = FermionicCircuit(6)  # norb = 3: alpha modes 0..3, beta modes 3..6
   >>> circuit.append(InitializeModes([1, 1, 0]), circuit.modes[:3])
   >>> circuit.append(OrbitalRotation(np.eye(3)), circuit.modes[:3])
   >>> circuit.append(InitializeModes([1, 0, 0]), circuit.modes[3:])
   >>> circuit.append(OrbitalRotation(np.eye(3)), circuit.modes[3:])
   >>> show_merged(circuit)
   PrepareSlaterDeterminant on modes [0, 1, 2]
   PrepareSlaterDeterminant on modes [3, 4, 5]

Global initialization, per-spin rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single full-register (``2 * norb``) :class:`.InitializeModes` followed by *two*
:class:`.OrbitalRotation`\ s, one on each contiguous spin half. The pass splits the global
occupation at the sector boundary and emits **two** :class:`.PrepareSlaterDeterminant` gates. The
two rotations may appear in either order.

This is the shape produced by the LUCJ workflow (see the :ref:`LUCJ guide <lucj_getting_started>`):
a user places an :class:`.InitializeModes` -- typically the Hartree-Fock reference via
:meth:`~qiskit_fermions.circuit.library.InitializeModes.from_hartree_fock` -- at the front of the
circuit and appends a :class:`.UCJ` ansatz, whose first two per-spin orbital rotations directly
follow the initialization once the ansatz is decomposed.

.. doctest::

   >>> circuit = FermionicCircuit(6)  # norb = 3, nelec = (2, 1)
   >>> circuit.append(InitializeModes.from_hartree_fock(3, (2, 1)), circuit.modes)
   >>> circuit.append(OrbitalRotation(np.eye(3)), circuit.modes[:3])
   >>> circuit.append(OrbitalRotation(np.eye(3)), circuit.modes[3:])
   >>> show_merged(circuit)
   PrepareSlaterDeterminant on modes [0, 1, 2]
   PrepareSlaterDeterminant on modes [3, 4, 5]

This pattern also fires when only *one* spin half is rotated by a gate directly following the
initialization -- the shape produced when just one spin sector's orbital rotation happens to sit at
the front of the circuit. The rotated half becomes a :class:`.PrepareSlaterDeterminant` with its
rotation, and the other half is prepared with an *identity* rotation. The identity preparation
synthesizes to nothing more than the reference X gates the :class:`.InitializeModes` would have
emitted for that half anyway, so padding it costs no extra gates while still unlocking the reduced
Slater synthesis on the rotated half:

.. doctest::

   >>> circuit = FermionicCircuit(6)
   >>> circuit.append(InitializeModes.from_hartree_fock(3, (2, 1)), circuit.modes)
   >>> circuit.append(OrbitalRotation(np.eye(3)), circuit.modes[:3])
   >>> show_merged(circuit)
   PrepareSlaterDeterminant on modes [0, 1, 2]
   PrepareSlaterDeterminant on modes [3, 4, 5]

What is left untouched
----------------------

The pass is deliberately conservative: it only fuses when the two gates genuinely compose into a
Slater determinant preparation, and copies everything else through unchanged. "Immediately
followed" is understood over the circuit's data-flow graph -- the :class:`.OrbitalRotation` fuses
only when the :class:`.InitializeModes` is its *sole* predecessor across all of its modes.

An operation in between
~~~~~~~~~~~~~~~~~~~~~~~~

If any gate touches the modes between the initialization and the rotation, they no longer compose
into a bare preparation, so nothing is merged:

.. doctest::

   >>> from qiskit_fermions.circuit.library import Evolution
   >>> from qiskit_fermions.operators import FermionOperator, ann, cre
   >>>
   >>> circuit = FermionicCircuit(4)
   >>> circuit.append(InitializeModes([1, 1, 0, 0]), circuit.modes)
   >>> number_op = FermionOperator.from_dict({(cre(0), ann(0)): 1.0})
   >>> circuit.append(Evolution(4, number_op, time=0.5), circuit.modes)
   >>> circuit.append(OrbitalRotation(np.eye(4)), circuit.modes)
   >>> show_merged(circuit)
   InitializeModes on modes [0, 1, 2, 3]
   Evolution on modes [0, 1, 2, 3]
   OrbitalRotation on modes [0, 1, 2, 3]

A partial rotation that is not a spin half
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A rotation covering exactly one contiguous spin half of a full-register initialization *does* fuse
(see the previous section). But a rotation on any *other* partial mode set -- fewer modes than the
initialization and not one of its two spin halves, such as a sub-range straddling the sector
boundary -- is not a per-sector preparation. Merging it would silently drop the initialization's
role on the modes it leaves out, so the pass leaves both gates in place:

.. doctest::

   >>> circuit = FermionicCircuit(4)  # norb = 2: the only spin halves are modes [0, 1] and [2, 3]
   >>> circuit.append(InitializeModes([1, 1, 0, 0]), circuit.modes)
   >>> circuit.append(OrbitalRotation(np.eye(3)), circuit.modes[:3])
   >>> show_merged(circuit)
   InitializeModes on modes [0, 1, 2, 3]
   OrbitalRotation on modes [0, 1, 2]

Why it matters: the synthesis payoff
-------------------------------------

The merge is worthwhile because :class:`.PrepareSlaterDeterminant` synthesizes to far fewer gates
than the :class:`.OrbitalRotation` it absorbs. An :math:`m`-electron determinant over :math:`n`
orbitals needs at most :math:`m(n-m)` two-qubit rotations and **no** diagonal phase gates, versus the
:math:`n(n-1)/2` rotations plus :math:`n` phases of the full square orbital rotation.

The preset Jordan-Wigner pass manager runs :class:`.MergeSlaterDeterminantPreparation` in its
optimization stage, so this reduction happens automatically. Here we transpile a two-electron,
six-orbital preparation all the way to a :class:`~qiskit.circuit.QuantumCircuit` and count its gates:

.. doctest::

   >>> from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
   >>>
   >>> # a reproducible random 6 x 6 orbital rotation
   >>> rng = np.random.default_rng(1)
   >>> z = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
   >>> q, r = np.linalg.qr(z)
   >>> rotation = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
   >>>
   >>> circuit = FermionicCircuit(6)
   >>> circuit.append(InitializeModes([1, 1, 0, 0, 0, 0]), circuit.modes)
   >>> circuit.append(OrbitalRotation(rotation), circuit.modes)
   >>>
   >>> pm = generate_preset_jw_pass_manager()
   >>> print(dict(sorted(pm.run(circuit).count_ops().items())))
   {'x': 2, 'xx_plus_yy': 8}

Compare that to synthesizing the same orbital rotation on its own -- with no preceding
initialization there is nothing to merge, so the preset falls back to the full square
decomposition, which is both deeper and carries diagonal phase gates:

.. doctest::

   >>> reference = FermionicCircuit(6)
   >>> reference.append(OrbitalRotation(rotation), reference.modes)
   >>> print(dict(sorted(pm.run(reference).count_ops().items())))
   {'p': 6, 'xx_plus_yy': 15}

The reduced decomposition drops all six phase gates and nearly halves the two-qubit rotation count.
The prepared determinant is correct up to a global phase -- physically irrelevant for state
preparation -- which is exactly what the phase gates would have fixed.

.. seealso::
   :class:`.PrepareSlaterDeterminant`, :class:`.InitializeModes`, :class:`.OrbitalRotation`,
   :class:`.GivensDecompositionSlaterDeterminantSynthesis`, and the
   :ref:`LUCJ guide <lucj_getting_started>` for the workflow that produces the global-initialization
   pattern.
