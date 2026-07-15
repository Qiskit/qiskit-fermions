.. _lucj_getting_started:

Build an LUCJ ansatz
====================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available through the C API in a future
   release.

The local unitary cluster Jastrow (`LUCJ`_) ansatz is a compact, hardware-efficient
parametrization of a correlated electronic wavefunction. It is a member of the more
general unitary cluster Jastrow (UCJ) family and takes the form

.. math::

   \lvert \Psi \rangle = \left(\prod_{k=1}^{L} \mathcal{U}_k\, e^{i \mathcal{J}_k}\,
   \mathcal{U}_k^\dagger\right) \lvert \Phi_0 \rangle,

where :math:`\lvert \Phi_0 \rangle` is a reference state (typically Hartree-Fock),
each :math:`\mathcal{U}_k` is an :ref:`orbital rotation <fermionic_circuit_explanation>`,
and each :math:`\mathcal{J}_k` is a diagonal Coulomb operator

.. math::

   \mathcal{J} = \frac12 \sum_{ij,\sigma\tau} \mathbf{J}^{\sigma\tau}_{ij}\,
   n_{i\sigma}\, n_{j\tau},

with :math:`n_{i\sigma}` the number operator on spatial orbital :math:`i` with spin
:math:`\sigma`. This guide shows how to assemble such an ansatz for a real molecule using
the fermionic gates in :mod:`qiskit_fermions.circuit.library`: an
:class:`.InitializeModes` gate for the reference state, :class:`.OrbitalRotation` gates for
the :math:`\mathcal{U}_k`, and :class:`.Evolution` gates for the :math:`e^{i \mathcal{J}_k}`
factors.

Both the ansatz parameters and the molecular Hamiltonian are built with this package's own API:
the ansatz layers come from :mod:`~qiskit_fermions.linalg`, and the Hamiltonian is assembled as a
:class:`.FermionOperator` from the active-space integrals. This guide uses
`PySCF <https://pyscf.org/>`_ to run the underlying quantum chemistry, and the optional ``ffsim``
dependency (managed by :data:`.HAS_FFSIM`) only to prepare and evolve the state vector.

.. invisible-code-block: python

   >>> from qiskit_fermions.utils.optionals import HAS_FFSIM
   >>> from qiskit.utils import LazyImportTester
   >>> HAS_PYSCF = LazyImportTester("pyscf", name="PySCF")

.. skip: start if(not HAS_FFSIM or not HAS_PYSCF)

1. Run the classical calculation and choose an active space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The (L)UCJ ansatz can be initialized from the amplitudes of a coupled-cluster singles and
doubles (CCSD) calculation. Here we run restricted Hartree-Fock for a nitrogen molecule in the
``sto-6g`` basis, then define an active space by freezing the two core orbitals. The CCSD
calculation that provides the :math:`t`-amplitudes is run with the same frozen core.

.. code-block:: python

   >>> import pyscf
   >>> import pyscf.cc
   >>>
   >>> # build the molecule and run Hartree-Fock
   >>> mol = pyscf.gto.Mole()
   >>> mol.build(
   ...     atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.1)]],
   ...     basis="sto-6g",
   ...     symmetry="Dooh",
   ...     verbose=0,
   ... )
   <pyscf.gto.mole.Mole object at ...>
   >>> scf = pyscf.scf.RHF(mol).run()
   >>>
   >>> # freeze the two core orbitals to define the active space
   >>> n_frozen = 2
   >>> active_space = range(n_frozen, mol.nao_nr())
   >>> norb = len(active_space)
   >>> n_active_elec = int(sum(scf.mo_occ[active_space]))
   >>> nelec = (n_active_elec // 2, n_active_elec // 2)
   >>>
   >>> # run CCSD (with the same frozen core) for the t-amplitudes
   >>> ccsd = pyscf.cc.CCSD(scf, frozen=range(n_frozen)).run()
   >>> t1, t2 = ccsd.t1, ccsd.t2

2. Build the molecular Hamiltonian as a fermionic operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will need the active-space Hamiltonian later to evaluate the ansatz energy. We build it
directly as a :class:`.FermionOperator` from the active-space integrals: PySCF's ``CASCI``
helper returns the one-body integrals ``h1e`` (together with the constant core energy ``ecore``,
which folds in the frozen-core and nuclear-repulsion contributions) and the two-body integrals
``h2e``. The electronic-integral constructors
:meth:`~qiskit_fermions.operators.FermionOperator.from_1body_tril_spin_sym` and
:meth:`~qiskit_fermions.operators.FermionOperator.from_2body_tril_spin_sym` expect the integrals
in packed (lower-triangular) `chemist` ordering, which is exactly what PySCF produces.

.. code-block:: python

   >>> import pyscf.mcscf
   >>> from pyscf import ao2mo, lib
   >>>
   >>> from qiskit_fermions.operators import FermionOperator
   >>>
   >>> # active-space integrals and the constant (frozen-core + nuclear) energy
   >>> cas = pyscf.mcscf.CASCI(scf, norb, nelec)
   >>> h1e, ecore = cas.get_h1eff()
   >>> h2e = cas.get_h2eff()
   >>>
   >>> # pack into the lower-triangular chemist-ordered layout the constructors expect
   >>> h1e_tril = lib.pack_tril(h1e)
   >>> h2e_tril = lib.pack_tril(ao2mo.restore(4, h2e, norb))
   >>>
   >>> hamiltonian = FermionOperator.from_1body_tril_spin_sym(
   ...     h1e_tril, norb
   ... ) + FermionOperator.from_2body_tril_spin_sym(h2e_tril, norb)

3. Build the LUCJ circuit
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.UCJ` gate assembles the ansatz directly from the coupled-cluster amplitudes. Its
:meth:`~qiskit_fermions.circuit.library.UCJ.from_t_amplitudes` constructor performs a *double
factorization* of the :math:`t_2` amplitudes (via
:func:`~qiskit_fermions.linalg.double_factorized_t2`) to obtain the per-layer diagonal Coulomb
matrices and orbital rotations, and derives an optional final orbital rotation from the
:math:`t_1` amplitudes.

.. code-block:: python

   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import UCJ
   >>>
   >>> ansatz = UCJ.from_t_amplitudes(nelec, t2, t1=t1)
   >>>
   >>> circuit = FermionicCircuit(2 * norb)
   >>> circuit.append(ansatz, circuit.modes)

.. skip: end

Decomposing the gate reveals its anatomy: an :class:`.InitializeModes` gate prepares the
Hartree-Fock reference determinant, and each ansatz layer contributes an orbital rotation
:math:`\mathcal{U}_k^\dagger`, then :math:`e^{i\mathcal{J}_k}` (an :class:`.Evolution` of the
diagonal Coulomb operator :math:`\mathcal{J}_k`), then :math:`\mathcal{U}_k`, with a final
orbital rotation at the end. The orbital rotations act per spin sector, so each is placed on the
alpha modes ``0..norb`` and the beta modes ``norb..2*norb`` independently.

The plot below illustrates this structure for a small two-orbital, single-repetition example
built directly from explicit tensors (a diagonal Coulomb matrix and an orbital rotation):

.. plot::
   :alt: The gates that a UCJ ansatz decomposes into.
   :context: close-figs
   :include-source:

   >>> import numpy as np
   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import UCJ
   >>>
   >>> example_diag_coulomb = np.array([[[[0.0, 0.5], [0.5, 0.0]], [[1.0, 0.2], [0.2, 1.0]]]])
   >>> example_rotations = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=complex)
   >>> example_ansatz = UCJ(2, (1, 1), example_diag_coulomb, example_rotations)
   >>>
   >>> example_circuit = FermionicCircuit(2 * 2)
   >>> example_circuit.append(example_ansatz, example_circuit.modes)
   >>> example_circuit.decompose().draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

4. Simulate the ansatz and evaluate its energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. skip: start if(not HAS_FFSIM or not HAS_PYSCF)


Because every gate in the circuit implements ``ffsim``'s ``SupportsApplyUnitary`` protocol,
the whole :class:`.FermionicCircuit` can be applied to a fixed particle-number state vector with
:func:`ffsim.apply_unitary`, starting from the Hartree-Fock reference. The
:class:`.FermionOperator` likewise implements ffsim's ``SupportsLinearOperator`` protocol, so we
can obtain a SciPy :class:`~scipy.sparse.linalg.LinearOperator` for it via
:func:`ffsim.linear_operator` and evaluate the ansatz energy as the expectation value of the
molecular Hamiltonian, adding back the constant core energy.

.. code-block:: python

   >>> import ffsim
   >>> import numpy as np
   >>>
   >>> reference = ffsim.hartree_fock_state(norb, nelec)
   >>> state = ffsim.apply_unitary(reference, circuit, norb=norb, nelec=nelec)
   >>>
   >>> linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
   >>> energy = np.vdot(state, linop @ state).real + ecore
   >>>
   >>> # the LUCJ energy improves on the Hartree-Fock reference
   >>> bool(energy < scf.e_tot)
   True

.. skip: end

.. note::
   The molecular Hamiltonian and the ansatz are built entirely with the core
   :mod:`qiskit_fermions` API -- :class:`.FermionOperator` (including its electronic-integral
   constructors) and the :class:`.UCJ` gate (whose
   :meth:`~qiskit_fermions.circuit.library.UCJ.from_t_amplitudes` uses the exact double
   factorization in :func:`~qiskit_fermions.linalg.double_factorized_t2`) -- plus NumPy and
   SciPy. Running the classical chemistry requires PySCF, and preparing and evolving the state
   vector (as well as wrapping the operator via :func:`ffsim.linear_operator`) require the
   optional dependency managed by :data:`.HAS_FFSIM`.

   To use ffsim's optimized ("compressed") double factorization instead, build an ``ffsim`` UCJ
   operator with ``optimize=True`` and pass its ``diag_coulomb_mats`` / ``orbital_rotations`` /
   ``final_orbital_rotation`` into :class:`.UCJ` directly.

Next steps
^^^^^^^^^^

- Learn how the individual gates work in the :mod:`qiskit_fermions.circuit.library`
  documentation and the :ref:`fermionic circuit guide <fermionic_circuit_explanation>`.
- Explore the :ref:`operators explanation guide <operators_explanation>` to understand how to
  construct fermionic Hamiltonians such as the diagonal Coulomb operator used above.
- See how a fermionic circuit is mapped to qubits in the
  :ref:`transpilation guide <transpilation_explanation>`.

.. _LUCJ: https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc02516k
