.. _lucj_getting_started:

Build an LUCJ ansatz
====================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available in the C API in a future
   release.

The local unitary cluster Jastrow (`LUCJ`_) ansatz is a compact, hardware-efficient
parametrization of a correlated electronic wavefunction. It is a member of the more
general unitary cluster Jastrow (UCJ) family and takes the form

.. math::

   \lvert \Psi \rangle = \left(\prod_{k=1}^{L} \mathcal{U}_k\, e^{i \mathcal{J}_k}\,
   \mathcal{U}_k^\dagger\right) \lvert \Phi_0 \rangle,

where :math:`\lvert \Phi_0 \rangle` is a reference state (typically Hartree-Fock),
each :math:`\mathcal{U}_k` is an orbital rotation, and each :math:`\mathcal{J}_k` is a diagonal
Coulomb operator

.. math::

   \mathcal{J} = \frac12 \sum_{ij,\sigma\tau} \mathbf{J}^{\sigma\tau}_{ij}\,
   n_{i\sigma}\, n_{j\tau},

with :math:`n_{i\sigma}` the number operator on spatial orbital :math:`i` with spin
:math:`\sigma`. This guide shows how to assemble such an ansatz for a real molecule using the
:class:`.UCJ` gate from :mod:`qiskit_fermions.circuit.library`.

1. Run the classical calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The (L)UCJ ansatz can be initialized from the amplitudes of a coupled-cluster singles and
doubles (CCSD) calculation. Run restricted Hartree-Fock followed by CCSD for a hydrogen
molecule in the ``6-31g`` basis, using `PySCF <https://pyscf.org/>`_ for the quantum chemistry.

.. invisible-code-block: python

   >>> from qiskit_fermions.utils.optionals import HAS_FFSIM

.. skip: start if(not HAS_FFSIM)

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import pyscf
   >>> import pyscf.cc
   >>>
   >>> # build the molecule and run Hartree-Fock
   >>> mol = pyscf.gto.Mole()
   >>> mol.build(
   ...     atom=[["H", (0, 0, 0)], ["H", (0, 0, 0.74)]],
   ...     basis="6-31g",
   ...     symmetry="Dooh",
   ...     verbose=0,
   ... )
   <pyscf.gto.mole.Mole object at ...>
   >>> scf = pyscf.scf.RHF(mol).run()
   >>>
   >>> mo_coeff = scf.mo_coeff
   >>> norb = mo_coeff.shape[1]
   >>> nelec = (mol.nelec[0], mol.nelec[1])
   >>>
   >>> # run CCSD for the t-amplitudes
   >>> ccsd = pyscf.cc.CCSD(scf).run()
   >>> t1, t2 = ccsd.t1, ccsd.t2

2. Build the molecular Hamiltonian as a fermionic operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Hamiltonian is needed later to evaluate the ansatz energy. Build it directly as a
:class:`.FermionOperator` from the molecular-orbital integrals: the one-body integrals ``h1e``
(the core Hamiltonian in the MO basis), the two-body integrals ``h2e`` (from :func:`pyscf.ao2mo`),
and the constant nuclear-repulsion energy. The electronic-integral constructors
:meth:`~qiskit_fermions.operators.FermionOperator.from_1body_tril_spin_sym` and
:meth:`~qiskit_fermions.operators.FermionOperator.from_2body_tril_spin_sym` expect the integrals
in packed (lower-triangular) `chemist` ordering, which is what PySCF produces.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from pyscf import ao2mo, lib
   >>>
   >>> from qiskit_fermions.operators import FermionOperator
   >>>
   >>> # one- and two-body molecular-orbital integrals and the nuclear-repulsion energy
   >>> h1e = mo_coeff.T @ scf.get_hcore() @ mo_coeff
   >>> h2e = ao2mo.kernel(mol, mo_coeff)
   >>> ecore = mol.energy_nuc()
   >>>
   >>> # pack into the lower-triangular chemist-ordered layout the constructors expect
   >>> h1e_tril = lib.pack_tril(h1e)
   >>> h2e_tril = lib.pack_tril(h2e)
   >>>
   >>> hamiltonian = ecore * FermionOperator.one()
   >>> hamiltonian += FermionOperator.from_1body_tril_spin_sym(h1e_tril, norb)
   >>> hamiltonian += FermionOperator.from_2body_tril_spin_sym(h2e_tril, norb)

3. Build the LUCJ circuit
^^^^^^^^^^^^^^^^^^^^^^^^^

The ansatz operator is built by `ffsim`_. Its
:external:meth:`~ffsim.UCJOpSpinBalanced.from_t_amplitudes` constructor performs a *double
factorization* of the :math:`t_2` amplitudes to obtain the per-layer diagonal Coulomb matrices and
orbital rotations, and derives an optional final orbital rotation from the :math:`t_1` amplitudes.
The :class:`.UCJ` gate then turns that operator into a fermionic circuit.

The number of ansatz repetitions, :math:`L`, equals the number of terms in the double
factorization. Truncating it with the ``n_reps`` argument trades some accuracy for a shallower
circuit; here the two largest terms are kept, which recovers most of the correlation energy while
halving the number of layers.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import ffsim
   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import InitializeModes, UCJ
   >>>
   >>> ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=2)
   >>> ansatz = UCJ(ucj_op)
   >>>
   >>> circuit = FermionicCircuit(2 * norb)
   >>> circuit.append(InitializeModes.from_hartree_fock(norb, nelec), circuit.modes)
   >>> circuit.append(ansatz, circuit.modes)

The :class:`.UCJ` gate is a pure unitary carrying no reference of its own, so prepend an
:class:`.InitializeModes` gate (built with
:meth:`~qiskit_fermions.circuit.library.InitializeModes.from_hartree_fock`) to supply the
Hartree-Fock reference the ansatz is applied to. Decomposing the circuit reveals its anatomy. The
:class:`.InitializeModes` gate prepares the reference determinant, and each ansatz layer contributes
an :class:`.OrbitalRotation` :math:`\mathcal{U}_k^\dagger`, then :math:`e^{i\mathcal{J}_k}` (an
:class:`.Evolution` of the diagonal Coulomb operator :math:`\mathcal{J}_k`), then
:math:`\mathcal{U}_k`, with a final :class:`.OrbitalRotation` at the end. The orbital rotations act
per spin sector, so each is placed on the alpha modes ``0..norb`` and the beta modes
``norb..2*norb`` independently.

.. plot::
   :alt: The gates that the UCJ ansatz decomposes into.
   :context: close-figs
   :include-source:

   >>> circuit.decompose().draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

.. note::
   Each layer ends with :math:`\mathcal{U}_k` and the next begins with
   :math:`\mathcal{U}_{k+1}^\dagger`, so adjacent :class:`.OrbitalRotation` gates could be merged
   into a single rotation. A transpilation pass performing this fusion is a planned future
   development.

4. Simulate the ansatz and evaluate its energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because every gate in the circuit implements ffsim's :class:`ffsim.SupportsApplyUnitary` protocol,
the whole :class:`.FermionicCircuit` can be applied to a fixed particle-number state vector with
:func:`ffsim.apply_unitary`, starting from the Hartree-Fock reference. The
:class:`.FermionOperator` likewise implements ffsim's :class:`ffsim.SupportsLinearOperator`
protocol, so you can obtain a SciPy :class:`~scipy.sparse.linalg.LinearOperator` for it via
:func:`ffsim.linear_operator` and evaluate the ansatz energy as the expectation value of the
molecular Hamiltonian.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import ffsim
   >>> import numpy as np
   >>>
   >>> reference = ffsim.hartree_fock_state(norb, nelec)
   >>> state = ffsim.apply_unitary(reference, circuit, norb=norb, nelec=nelec)
   >>>
   >>> linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
   >>> energy = np.vdot(state, linop @ state).real
   >>> print(f"LUCJ energy: {energy:.8f} Hartree")
   LUCJ energy: -1.14618323 Hartree

The LUCJ energy improves substantially on the Hartree-Fock reference and approaches the CCSD
energy it was initialized from; the small remaining gap is the price of truncating the ansatz to
two repetitions:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> print(f"Hartree-Fock: {scf.e_tot:.8f} Hartree")
   Hartree-Fock: -1.12675532 Hartree
   >>> print(f"CCSD:         {ccsd.e_tot:.8f} Hartree")
   CCSD:         -1.15167268 Hartree

.. skip: end

5. (Optional) Use ffsim's compressed double factorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.UCJ` gate above is initialized from an *exact* double factorization of the :math:`t_2`
amplitudes. The number of ansatz repetitions :math:`L` is whatever that factorization yields (up to
the ``n_reps`` truncation), and each layer reproduces one factorized term exactly. `ffsim`_
additionally offers an optimized ("compressed") double factorization, its
``from_t_amplitudes(..., optimize=True)``, which variationally fits the amplitudes with a chosen,
typically smaller, number of repetitions. This trades a classical optimization up front for a
shallower ansatz at a target accuracy, and has no equivalent in this package.

Since :class:`.UCJ` takes the ffsim operator itself, this needs no extra work: pass
``optimize=True`` when building the operator and hand the result to the same constructor.

.. skip: start if(not HAS_FFSIM)

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> compressed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
   ...     t2, t1=t1, n_reps=2, optimize=True
   ... )
   >>>
   >>> compressed_ansatz = UCJ(compressed)
   >>>
   >>> compressed_circuit = FermionicCircuit(2 * norb)
   >>> compressed_circuit.append(
   ...     InitializeModes.from_hartree_fock(norb, nelec), compressed_circuit.modes
   ... )
   >>> compressed_circuit.append(compressed_ansatz, compressed_circuit.modes)

The resulting circuit is used like the one built from the exact factorization (the
:class:`.UCJ` gate does not care how its tensors were obtained), and evaluating its energy the same
way recovers the same correlation energy at this (small) system size:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> state = ffsim.apply_unitary(reference, compressed_circuit, norb=norb, nelec=nelec)
   >>> energy = np.vdot(state, linop @ state).real
   >>> print(f"compressed LUCJ energy: {energy:.8f} Hartree")
   compressed LUCJ energy: -1.14618323 Hartree

.. note::
   For a better fit at a given ``n_reps`` (at increased classical cost) see ffsim's
   ``multi_stage_start`` / ``multi_stage_step`` options.

.. skip: end

6. Transpile the ansatz to a qubit circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run the ansatz on hardware, it must be lowered from fermionic modes to qubits.
The :func:`~qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager` preset builds a
staged pipeline that maps the fermionic circuit through the Jordan-Wigner transformation and
synthesizes each gate into a qubit-level circuit. The composite :class:`.UCJ` gate must first be
decomposed into its primitive gates (:class:`.OrbitalRotation`, :class:`.Evolution`, ...) so the
pipeline's optimization stage can act on them, so pass ``circuit.decompose()``.

.. skip: start if(not HAS_FFSIM)

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
   >>>
   >>> # ``circuit`` is the exact-factorization ansatz assembled in step 3
   >>> pm = generate_preset_jw_pass_manager()
   >>> transpiled = pm.run(circuit.decompose())
   >>> print(dict(sorted(transpiled.count_ops().items())))
   {'p': 14, 'rzz': 12, 'x': 2, 'xx_plus_yy': 28}

Without a target device, this maps onto ``2 * norb`` qubits with all-to-all connectivity assumed; the
orbital rotations synthesize into :class:`~qiskit.circuit.library.XXPlusYYGate`\ objects and the diagonal
Coulomb evolutions into :class:`~qiskit.circuit.library.RZZGate`\ objects:

.. plot::
   :alt: The Jordan-Wigner transpiled LUCJ circuit.
   :context: close-figs
   :include-source:

   >>> transpiled.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

.. rubric:: Target hardware connectivity with ffsim's LUCJ pass manager

A real device has a restricted qubit coupling map, and the LUCJ ansatz is designed to match it. The
same-spin (``pairs_aa``) interactions form two linear chains and the alpha-beta (``pairs_ab``)
interactions bridge them. ffsim's :external:func:`~ffsim.qiskit.generate_lucj_pass_manager` builds a
device-aware qubit pipeline for this structure, and returns the subset of ``pairs_ab`` the
hardware can actually accommodate. Slot that pipeline into the preset's ``qubit`` stage while
keeping the package's own fermion-to-qubit synthesis:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from ffsim.qiskit import generate_lucj_pass_manager
   >>> from qiskit.providers.fake_provider import GenericBackendV2
   >>> from qiskit.transpiler import CouplingMap
   >>>
   >>> # a heavy-hex device coupling map (any BackendV2 works, e.g. a real fake_provider backend)
   >>> coupling_map = CouplingMap.from_heavy_hex(5)
   >>> backend = GenericBackendV2(
   ...     num_qubits=coupling_map.size(),
   ...     basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
   ...     coupling_map=coupling_map,
   ... )
   >>>
   >>> # nearest-neighbor same-spin chain; let the pass manager choose the alpha-beta pairs
   >>> pairs_aa = [(p, p + 1) for p in range(norb - 1)]
   >>>
   >>> pm = generate_preset_jw_pass_manager()
   >>> pm.qubit, allowed_pairs_ab = generate_lucj_pass_manager(
   ...     backend, norb, "heavy-hex", (pairs_aa, None), optimization_level=3, seed_transpiler=0
   ... )
   >>>
   >>> # the alpha-beta interactions the heavy-hex connectivity can implement
   >>> print(allowed_pairs_ab)
   [(0, 0)]

With ``pm.qubit`` now set to the device-aware pipeline, running the pass manager lays the circuit out
on the backend's qubits and routes it to the coupling map. For the *unrestricted* ansatz from
step 3, whose diagonal Coulomb operator still contains alpha-beta terms the hardware cannot reach
directly, the router must insert many ``SWAP`` gates to bridge them:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> naive = pm.run(circuit.decompose())
   >>> naive.num_qubits  # laid out on the full heavy-hex device register
   57
   >>> naive_swaps = naive.count_ops()["swap"]
   >>> naive_swaps  # many SWAPs to bridge the unreachable alpha-beta interactions
   33

.. rubric:: Restrict the ansatz to the hardware-implementable interactions

The fix is to feed ``allowed_pairs_ab`` back into the ansatz construction, through the
``interaction_pairs`` argument of
:external:meth:`~ffsim.UCJOpSpinBalanced.from_t_amplitudes`, so the diagonal Coulomb operator only
contains alpha-beta terms the coupling map can implement directly. The ansatz then matches the device topology and the router barely has to touch it:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> restricted = UCJ(
   ...     ffsim.UCJOpSpinBalanced.from_t_amplitudes(
   ...         t2, t1=t1, n_reps=2, interaction_pairs=(pairs_aa, allowed_pairs_ab)
   ...     )
   ... )
   >>>
   >>> circuit = FermionicCircuit(2 * norb)
   >>> circuit.append(InitializeModes.from_hartree_fock(norb, nelec), circuit.modes)
   >>> circuit.append(restricted, circuit.modes)
   >>>
   >>> transpiled = pm.run(circuit.decompose())
   >>> restricted_swaps = transpiled.count_ops()["swap"]
   >>> restricted_swaps  # far fewer routing SWAPs than the unrestricted ansatz
   4

Drawing only the active qubits (``idle_wires=False``) shows the circuit restricted to the two spin
chains and the alpha-beta bridge, expressed in the device basis gates. Layout and routing scatter the
logical modes across the device's physical qubits, so pass a ``wire_order`` taken from the
circuit's final layout (:meth:`~qiskit.transpiler.TranspileLayout.final_index_layout` lists the
physical qubit each input qubit ended on, in input-qubit order) to draw the wires back in the
original mode order:

.. plot::
   :alt: The hardware-restricted LUCJ circuit routed onto the heavy-hex device coupling map.
   :context: close-figs
   :include-source:

   >>> wire_order = transpiled.layout.final_index_layout(filter_ancillas=False)
   >>> transpiled.draw("mpl", idle_wires=False, fold=-1, wire_order=wire_order)
   <Figure size ... with 1 Axes>

.. note::
   The exact post-layout gate counts and depth depend on the routing/optimization passes and the
   chosen device, so they are not reproduced here. The key point is the co-design; expressing the
   ansatz with a nearest-neighbor ``pairs_aa`` chain and hardware-filtered ``pairs_ab`` bridges keeps
   the synthesized circuit close to the device topology, minimizing the routing overhead (inserted
   ``SWAP`` gates). See :class:`.GivensDecompositionSlaterDeterminantSynthesis` for a related
   synthesis choice (``minimize_2q_gate_count``), trading two-qubit gate count against routed depth.

.. skip: end

Next steps
^^^^^^^^^^

- Learn how the individual gates work in the :mod:`qiskit_fermions.circuit.library`
  documentation and the :ref:`fermionic circuit guide <fermionic_circuit_explanation>`.
- Explore the :ref:`operators explanation guide <operators_explanation>` to understand how to
  construct fermionic Hamiltonians such as the diagonal Coulomb operator used above.
- See how a fermionic circuit is mapped to qubits in the
  :ref:`transpilation guide <transpilation_explanation>`.
- Read the :ref:`ffsim backend guide <ffsim_backend_explanation>` to understand why
  :func:`ffsim.apply_unitary` and :func:`ffsim.linear_operator` work natively on this ansatz, and how
  to evaluate its energy without ffsim installed.

.. _LUCJ: https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc02516k
.. _ffsim: https://qiskit-community.github.io/ffsim/
