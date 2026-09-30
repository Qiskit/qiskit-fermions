.. _fermionic_circuit_explanation:

Work with fermionic circuits
============================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available in the C API in a future
   release.

This guide explains how to use the :class:`.FermionicCircuit` to implement
quantum algorithms directly in fermionic space, rather than mapping to qubits
first.

Why use fermionic circuits
--------------------------

Consider the traditional workflow for implementing the time evolution of a
fermionic Hamiltonian on a qubit-based architecture:

1. Define the fermionic Hamiltonian (for example, :math:`H = \sum_{pq} h_{pq}
   c^\dagger_p c_q`).
2. Map it to a qubit operator (for example, by using the Jordan-Wigner mapping).
3. Implement the time evolution in qubit space (using Trotterization or other
   methods).

This approach has a fundamental limitation; early mapping to qubits discards
problem-aware knowledge about the fermionic structure. Standard optimization
passes cannot exploit fermionic symmetries or allowed term reorderings,
resulting in suboptimal circuits with unnecessary depths and gate counts.

The :class:`.FermionicCircuit` enables a better workflow:

1. Define the fermionic Hamiltonian (for example, :math:`H = \sum_{pq} h_{pq}
   c^\dagger_p c_q`).
2. Implement the time evolution directly in fermionic space using fermionic
   gates.
3. Map to qubit space as part of the transpilation process.

This approach addresses the limitation by keeping problem-aware knowledge in
fermionic space, where optimization passes can exploit fermionic structure and
commutation relations to reduce circuit depth. The fermionic circuit describes
what computation to perform, while transpilation handles how to map it to
qubits, providing a cleaner separation of concerns.

Generic mode indexing
---------------------

Both the :mod:`~qiskit_fermions.operators` module and the
:class:`.FermionicCircuit` use generic mode-based indexing that makes no
assumptions about the nature of the modes. A mode is simply an abstract index
labeling a fermionic degree of freedom, with no inherent semantics. This design
choice mirrors the operators module and ensures maximum flexibility.
For example, you can implement time evolution for any operator implementing the
:class:`.OperatorTrait` protocol, regardless of its mathematical representation.

Build a fermionic circuit
-------------------------

The example below implements the time evolution described in the previous section.
It constructs a :class:`.FermionicCircuit` by specifying the number of fermionic
modes, then adds fermionic gates from the :mod:`qiskit_fermions.circuit.library`
to implement the time evolution.

The example also demonstrates how to incorporate domain knowledge into an
operator by using :ref:`operator term grouping <grouping_explanation>`. By
assigning group indices to the Hamiltonian terms, you preserve structural
information that optimization passes can exploit throughout the transpilation
stack.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import Evolution
   >>> from qiskit_fermions.operators import FermionOperator, cre, ann
   >>>
   >>> # Create a circuit with 4 fermionic modes
   >>> circuit = FermionicCircuit(4)
   >>>
   >>> # Define a simple fermionic Hamiltonian
   >>> hamiltonian = FermionOperator.from_terms([
   ...     ([cre(0), ann(2)], 0.5),
   ...     ([cre(2), ann(0)], 0.5),
   ...     ([cre(1), ann(3)], 0.5),
   ...     ([cre(3), ann(1)], 0.5),
   ... ])
   >>> # Add some problem structure by grouping our Hamiltonian terms
   >>> hamiltonian.groups = [0, 0, 1, 1]
   >>>
   >>> # Add an evolution gate to implement exp(-i * t * H)
   >>> evolution = Evolution(4, hamiltonian, time=1.0)
   >>> circuit.append(evolution, circuit.register)

.. plot::
   :alt: A simple `FermionicCircuit` with a single time evolution gate.
   :context: close-figs

   >>> circuit.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

.. plot::
   :alt: A decomposed `FermionicCircuit` with several time evolution gates.
   :context: close-figs

   >>> circuit.decompose().draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

Notice how the operator term grouping is preserved even in simple operations
like decomposition. This demonstrates how structural information flows through the circuit
stack. To understand the full transpilation to
qubits, refer to the :ref:`Transpiling fermionic circuits <transpilation_explanation>` guide.

Keep gates apart with a barrier
-------------------------------

The optimization passes fuse neighboring gates where they can. For example,
:class:`.MergeOrbitalRotations` combines a run of consecutive
:class:`.OrbitalRotation` gates into one, and
:class:`.MergeSlaterDeterminantPreparation` folds an :class:`.InitializeModes`
into the rotation that follows it.

Use :meth:`.FermionicCircuit.barrier` when you want to prevent that. A barrier
carries no unitary effect, so it does not change the state your circuit
prepares; it only marks a point that the passes do not fuse across. Call it
without arguments to span every mode, or pass the modes to restrict it to:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.circuit.library import OrbitalRotation
   >>> import numpy as np
   >>>
   >>> rotations = FermionicCircuit(4)
   >>> rotations.append(OrbitalRotation(np.eye(4, dtype=complex)), rotations.register)
   >>> rotations.barrier()
   >>> rotations.append(OrbitalRotation(np.eye(4, dtype=complex)), rotations.register)

.. plot::
   :alt: A `FermionicCircuit` with a barrier separating two orbital rotations.
   :context: close-figs

   >>> rotations.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

Barriers survive the fermion-to-qubit transpilation, so they also constrain the
qubit-level optimization that follows. The
:ref:`Transpile fermionic circuits <transpilation_explanation>` guide covers
that stage.

Repeat a circuit
----------------

Many algorithms apply the same block of gates many times over. Time evolution is
the common case: a Trotter step is built once, then repeated to reach the target
time. Use :meth:`~.FermionicCircuit.repeat` to build the repeated circuit, and
:meth:`~.FermionicCircuit.compose` to join it to a state preparation prefix.

The example below builds a second-order Trotter step for a Fermi-Hubbard model.
The step is symmetric: it splits the hopping term into two half-duration
:class:`.OrbitalRotation` gates surrounding the on-site interaction.

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> import scipy.linalg
   >>> from qiskit_fermions.circuit.library import InitializeModes, OrbitalRotation
   >>>
   >>> norb, reps, time = 2, 3, 1.0
   >>> num_modes = 2 * norb
   >>> dt = time / reps
   >>>
   >>> hopping = np.zeros((norb, norb))
   >>> hopping[0, 1] = hopping[1, 0] = -1.0
   >>> interaction = FermionOperator.from_dict(
   ...     {(cre(p), ann(p), cre(p + norb), ann(p + norb)): 2.0 for p in range(norb)}
   ... )
   >>>
   >>> def rotation(duration):
   ...     propagator = scipy.linalg.expm(-1j * duration * hopping)
   ...     # the same rotation acts on the alpha and beta halves of the register
   ...     return scipy.linalg.block_diag(propagator, propagator)
   >>>
   >>> trotter_step = FermionicCircuit(num_modes)
   >>> trotter_step.append(OrbitalRotation(rotation(dt / 2)), trotter_step.modes)
   >>> trotter_step.append(
   ...     Evolution(num_modes, interaction, dt, atomic=True), trotter_step.modes
   ... )
   >>> trotter_step.append(OrbitalRotation(rotation(dt / 2)), trotter_step.modes)

Repeating the step and composing it onto a Hartree-Fock reference gives the full
evolution:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> evolution = FermionicCircuit(num_modes)
   >>> evolution.append(InitializeModes([True, False, True, False]), evolution.modes)
   >>> evolution.compose(trotter_step.repeat(reps), inplace=True)
   >>> evolution.count_ops()["OrbitalRotation"]
   6

.. plot::
   :alt: A `FermionicCircuit` with repeated steps.
   :context: close-figs

   >>> evolution.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

The repetitions are flattened into the returned circuit rather than wrapped in a
single gate, so :meth:`~.FermionicCircuit.count_ops` counts two rotations per
step. That flattening is what makes the redundancy at the step boundaries
visible to the transpiler: the trailing half-duration rotation of one step and
the leading one of the next are adjacent, so
:class:`.MergeOrbitalRotations` combines them into a single rotation.

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.transpiler import FermionicPassManager
   >>> from qiskit_fermions.transpiler.passes import MergeOrbitalRotations
   >>>
   >>> merged = FermionicPassManager([MergeOrbitalRotations()]).run(evolution)
   >>> merged.count_ops()["OrbitalRotation"]
   4

.. plot::
   :alt: A `FermionicCircuit` with repeated steps and merged orbital rotations.
   :context: close-figs

   >>> merged.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

The result costs one rotation per step plus one, which is what you get by fusing
those half-steps by hand. Writing the step in its natural symmetric form and
repeating it produces the same circuit, so the fusion is left to the
optimization stage.

When you need the repetitions kept apart, pass ``insert_barriers=True`` to place
a :meth:`~.FermionicCircuit.barrier` between them. The optimization passes do
not fuse gates across a barrier, so every rotation survives:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> separated = trotter_step.repeat(reps, insert_barriers=True)
   >>> FermionicPassManager([MergeOrbitalRotations()]).run(separated).count_ops()[
   ...     "OrbitalRotation"
   ... ]
   6

.. plot::
   :alt: A `FermionicCircuit` with repeated steps separated by barriers.
   :context: close-figs

   >>> separated.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

.. note::
   The ``reps`` argument of :class:`.FermionicSuzukiTrotter` is a different
   concept. It subdivides a `single` :class:`.Evolution` gate into more, shorter
   factors, whereas :meth:`~.FermionicCircuit.repeat` duplicates whole circuits.
   Repeating a circuit that holds one :class:`.Evolution` only multiplies its
   total evolution time.

Transpile fermionic circuits
----------------------------

To implement the quantum algorithm represented by your
:class:`.FermionicCircuit` it must be transpiled to a
:class:`~qiskit.circuit.QuantumCircuit`. You can learn how to do this in the
:ref:`Transpile fermionic circuits <transpilation_explanation>` guide.

Next steps
----------

- Learn about available fermionic gates in the :mod:`qiskit_fermions.circuit.library`
  documentation.
- Explore the :ref:`operators explanation guide <operators_explanation>` to understand
  how to construct fermionic Hamiltonians that you can use with fermionic circuits.
- Review the :ref:`SQDRIFT <sqdrift_getting_started>` getting started guide for a practical
  end-to-end example.
- Read the :ref:`ffsim relationship guide <ffsim_relationship_explanation>` to understand how a
  :class:`.FermionicCircuit` can be simulated directly, and how these generic mode indices take on
  spin semantics only at simulation time.
