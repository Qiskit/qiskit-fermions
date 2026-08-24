.. _ffsim_backend_explanation:

The ffsim simulation backend
============================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available in the C API in a future
   release.

The :ref:`getting-started guides <lucj_getting_started>` simulate fermionic circuits and
Hamiltonians with `ffsim`_, calling :func:`ffsim.apply_unitary`, :func:`ffsim.linear_operator`, or
:func:`ffsim.sample_state_vector` directly on :class:`.FermionicCircuit`\ objects and
:class:`.FermionOperator`\ objects. This guide explains why that works and what is happening
underneath. This package couples deliberately with ffsim's simulation protocols rather than
building its own simulation API, and it does so in a way that keeps a native (scipy-only)
simulation path available for users who cannot or do not want to install ffsim.

Why couple with ffsim
---------------------

`ffsim`_ is a high-performance simulator for fermionic quantum circuits that exploits
particle number and spin-Z conservation to represent state vectors more compactly than a
generic :math:`2^n`-dimensional qubit statevector. It defines two protocols that any object
can implement to participate in its simulation machinery (see :mod:`qiskit_fermions.protocols`
for this package's own protocols, which follow the same design):

- :class:`ffsim.SupportsApplyUnitary`, by the method ``_apply_unitary_(vec, norb, nelec, copy)``,
  applying the object as a unitary to a fixed-particle-number state vector
- :class:`ffsim.SupportsLinearOperator`, by the method ``_linear_operator_(norb, nelec)``, returning a
  :class:`scipy.sparse.linalg.LinearOperator` view of the object on that same sector.

Rather than invent a separate simulation interface, every fermionic gate in
:mod:`qiskit_fermions.circuit.library` and :class:`.FermionOperator` implement these 
protocols. The direct payoff is that ffsim's tools work natively on this package's objects,
with no conversion step, :func:`ffsim.apply_unitary` can simulate a :class:`.FermionicCircuit`
end to end, :func:`ffsim.linear_operator` (or plain :func:`scipy.sparse.linalg.eigsh`) can
diagonalize a :class:`.FermionOperator`, and sampling utilities like
:func:`ffsim.sample_state_vector` (used in the :ref:`SKQD guide <skqd_getting_started>` to turn a
simulated statevector into measurement counts) work without modification.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import numpy as np
   >>>
   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import Evolution
   >>> from qiskit_fermions.operators import FermionOperator, ann, cre
   >>> from qiskit_fermions.utils.optionals import HAS_FFSIM
   >>>
   >>> if HAS_FFSIM:
   ...     import ffsim
   >>>
   >>> norb, nelec = 2, (1, 1)
   >>> hamiltonian = FermionOperator.from_terms([
   ...     ([cre(0), ann(1)], 0.5),
   ...     ([cre(1), ann(0)], 0.5),
   ... ])
   >>>
   >>> circuit = FermionicCircuit(2 * norb)
   >>> circuit.append(Evolution(2 * norb, hamiltonian, time=1.0), circuit.modes)
   >>>
   >>> if HAS_FFSIM:
   ...     reference = ffsim.hartree_fock_state(norb, nelec)
   ...     state = ffsim.apply_unitary(reference, circuit, norb=norb, nelec=nelec)  # native ffsim call

This is the reason for coupling with ffsim's protocols rather than only offering a
bespoke ``simulate()`` method. It lets ffsim's existing, actively developed ecosystem of simulation
and sampling utilities apply to this package's circuits and operators unchanged, and it lets users
already working with ffsim mix in this package's gates and operators without learning a second
simulation API.

A native path when ffsim is unavailable
---------------------------------------

Coupling with ffsim's protocols does not make ffsim a hard dependency. `ffsim`_ transitively
depends on `PySCF <https://pyscf.org/>`_, which does not support Windows, so ffsim is declared as
an optional extra (``pip install "qiskit-fermions[simulation]"``, or transitively by ``[all]``),
guarded at runtime by :data:`~qiskit_fermions.utils.optionals.HAS_FFSIM` (as was shown above).
On Windows, that extra resolves to nothing (a silent no-op through a ``sys_platform`` marker),
rather than an install failure.

Simulation must still work without ffsim, so :class:`.SupportsLinearOperator` is backed
by an independent native Rust FCI (full configuration interaction) kernel, not a wrapper
around ffsim's linear algebra routines. It compiles an operator's terms into a scatter map
over the fixed-particle-number determinant basis, then reuses that compiled form across repeated
matrix-vector products. This is the protocol method ffsim expects
(:class:`ffsim.SupportsLinearOperator`): it is implemented without ffsim, and it is
cross-checked against ffsim's matrix elements in this package's test suite, but it does not call
into ffsim. Handing a :class:`.FermionOperator` to :func:`scipy.sparse.linalg.eigsh` or
:func:`scipy.sparse.linalg.expm_multiply` (as :meth:`.Evolution._apply_unitary_placed_` does
internally) therefore works identically whether or not ffsim is installed:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import scipy.sparse.linalg
   >>> from qiskit_fermions.linalg import linear_operator
   >>>
   >>> linop = linear_operator(hamiltonian, norb, nelec)  # pure scipy + native Rust kernel, no ffsim
   >>> energy, _ = scipy.sparse.linalg.eigsh(linop, k=1, which="SA")
   >>> print(f"ground-state energy: {energy[0]:.6f}")
   ground-state energy: -0.500000

Some gates go further and branch on :data:`~qiskit_fermions.utils.optionals.HAS_FFSIM` at simulation
time for performance. :class:`.OrbitalRotation` is one example, whose ``_apply_unitary_``
implementation delegates to ffsim's dedicated Givens-rotation kernel
(:func:`ffsim.apply_orbital_rotation`) when ffsim is installed, and otherwise falls back to
expressing the rotation as :math:`\exp(G)` for a one-body generator :math:`G` and applying that
through the same native-kernel-plus-:func:`~scipy.sparse.linalg.expm_multiply` path used throughout
this section. Both paths implement the same protocol method and produce the same result; ffsim is
a performance choice, not a correctness dependency. Gates without such a fast path
(:class:`.Evolution`, and everything built out of it, such as :class:`.UCJ`) use the
ffsim-independent path unconditionally.

In short: ffsim's protocols are the interface; ffsim is an accelerator you can uninstall.
This is also why the two getting started guides that use ffsim directly (:ref:`LUCJ
<lucj_getting_started>` and :ref:`SKQD <skqd_getting_started>`) internally guard their ffsim-specific
code with :data:`~qiskit_fermions.utils.optionals.HAS_FFSIM` the same way this guide does.

Fermionic simulation lives in a fixed particle-number sector
------------------------------------------------------------

Both ``_apply_unitary_`` and ``_linear_operator_`` take an ``nelec`` argument and represent the state
vector over the *fixed-particle-number determinant basis* for that ``(norb, nelec)`` sector,
mirroring ffsim's (and, transitively, PySCF's) FCI space setup, rather than the full
:math:`2^{\text{num\_modes}}`-dimensional space a general qubit simulator would use. This is a much
smaller space (its size is a product of binomial coefficients rather than a power of two), but it
comes with a hard restriction: only operators and gates that preserve particle number (and, in
the spinful case, each spin species' particle number individually) can be represented in it. An
operator whose action would move amplitude to a different particle number has nowhere to go
in this fixed-sector situation.

The native kernel resolves this by silently dropping any term that would leave the sector, projecting
its contribution to zero rather than raising an error, since the kernel's contract is "matrix-vector
product on this sector," not "validate this operator." For most callers this dropping would be the
wrong thing: applying :math:`\exp(-i t H)` for a Hamiltonian :math:`H` with a
non-particle-conserving term would silently turn a would-be unitary into a non-unitary,
physically-meaningless map. So the higher-level entry points that build a unitary out of the kernel
add an explicit guard before calling it:

- :class:`.Evolution` checks :meth:`.FermionOperator.conserves_sector` and raises a
  :class:`ValueError` if the operator does not conserve the ``(norb, nelec)`` sector.
- :class:`.OrbitalRotation` checks that its (embedded) rotation matrix is block-diagonal across the
  alpha/beta split in the spinful case, and raises a :class:`ValueError` if it mixes spin sectors.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.circuit.library import Evolution
   >>> from qiskit_fermions.linalg import apply_unitary
   >>>
   >>> non_conserving = FermionOperator.from_terms([([cre(0), cre(1)], 1.0)])  # creates 2 particles
   >>> gate = Evolution(2 * norb, non_conserving, time=1.0)
   >>>
   >>> if not HAS_FFSIM:
   ...     # on Windows we manually define our reference state vector
   ...     reference = np.asarray([1, 0, 0, 0], dtype=complex)
   >>>
   >>> try:
   ...     apply_unitary(reference, gate, norb, nelec, copy=True)
   ... except ValueError as exc:
   ...     print("rejected:", exc)
   rejected: Evolution requires an operator that conserves the (norb, nelec) sector: every term must preserve the particle number of each spin species (norb=2, nelec=(1, 1)).

Calling :meth:`.SupportsLinearOperator._linear_operator_` *directly* on a non-conserving operator bypasses
this guard (it is a lower-level building block, not a validated simulation entry point) and returns
a matrix-vector product that silently zeroes the non-conserving amplitude rather than raising.

The practical consequence is that non-particle-preserving simulation is not possible in fermionic
space, by construction of the fixed-sector representation: not as a missing feature, but
as the price of the compact FCI-space representation that makes fermionic simulation tractable. If your algorithm needs particle-number-violating operators (for example,
a qubit-native error channel, or an operator built for a mapped Hamiltonian that does not conserve
particle numbers term by term), transpile to qubits first and simulate the resulting
:class:`~qiskit.circuit.QuantumCircuit` with a qubit-level simulator instead. Any transpilation route works for this; the
:ref:`fermionic circuit <fermionic_circuit_explanation>` and :ref:`transpilation
<transpilation_explanation>` guides go into more detail, but
:func:`~qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager` is a reasonable default
choice.

The block-spin convention for spinful systems
---------------------------------------------

``nelec`` is typed ``int | tuple[int, int]``, and its type, not a separate flag, is what selects
one of the two supported mode layouts:

- An **int** selects the **spinless** interpretation: the ``norb`` modes are treated directly as
  spinless orbitals, and ``nelec`` is the total particle count.
- A **pair** ``(n_alpha, n_beta)`` selects the **spinful** interpretation of ``2 * norb`` modes under
  a fixed **block-spin convention**: modes ``0 .. norb`` are the alpha (spin-up) orbitals and modes
  ``norb .. 2 * norb`` are the beta (spin-down) orbitals. Each spin species is conserved
  *independently*: an alpha-only term can move an electron between alpha modes but never into a
  beta mode, and vice versa.

This dispatch happens at every ffsim-protocol entry point in this package (gate constructors,
``_apply_unitary_placed_`` implementations, and the native Rust kernel's sector compilation),
and it is the convention used throughout the :ref:`LUCJ <lucj_getting_started>` and
:ref:`SKQD <skqd_getting_started>` guides (for example,
:meth:`.InitializeModes.from_hartree_fock` fills alpha occupations into modes ``0 .. n_alpha`` and
beta occupations into modes ``norb .. norb + n_beta``). It is worth contrasting this with the
:mod:`qiskit_fermions.operators` module and :class:`.FermionicCircuit` in general, which use
*generic* mode indices with no inherent spin semantics (see the :ref:`fermionic circuit guide
<fermionic_circuit_explanation>`): the block-spin meaning is imposed only when a spinful ``nelec``
is supplied to a simulation call, not baked into the operator or circuit representation itself.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.circuit.library import InitializeModes
   >>>
   >>> norb, nelec = 3, (2, 1)
   >>> init = InitializeModes.from_hartree_fock(norb, nelec)
   >>> print([bool(occ) for occ in init.occupation])  # alpha modes 0,1; beta mode 3 (= norb + 0)
   [True, True, False, True, False, False]

Next steps
^^^^^^^^^^

- Walk through the :ref:`LUCJ <lucj_getting_started>` guide to see this backend used end to end,
  including the pure-scipy path (through the :func:`ffsim.linear_operator`, itself backed by the same
  ``_linear_operator_`` protocol method described here) for evaluating an ansatz's energy.
- Walk through the :ref:`SKQD <skqd_getting_started>` guide to see :func:`ffsim.apply_unitary` and
  :func:`ffsim.sample_state_vector` used together to sample circuits built from this package's gates.
- Read the :ref:`fermionic circuit guide <fermionic_circuit_explanation>` for the generic,
  spin-agnostic mode indexing used outside of simulation calls.
- Read the :ref:`transpilation guide <transpilation_explanation>` for how to leave fermionic space
   and simulate on qubits, which is required for non-particle-conserving operators.
- Browse :mod:`qiskit_fermions.protocols` for the full catalog of protocols this package defines,
  including the conversion protocols (:class:`.SupportsFermionOperator`,
  :class:`.SupportsMajoranaOperator`) that are unrelated to ffsim.

.. _ffsim: https://qiskit-community.github.io/ffsim/
