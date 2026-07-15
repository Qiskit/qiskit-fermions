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
doubles (CCSD) calculation. Here we run restricted Hartree-Fock followed by CCSD for a hydrogen
molecule in the ``6-31g`` basis, using `PySCF <https://pyscf.org/>`_ for the quantum chemistry.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will need the Hamiltonian later to evaluate the ansatz energy. We build it directly as a
:class:`.FermionOperator` from the molecular-orbital integrals: the one-body integrals ``h1e``
(the core Hamiltonian in the MO basis), the two-body integrals ``h2e`` (from :func:`pyscf.ao2mo`),
and the constant nuclear-repulsion energy. The electronic-integral constructors
:meth:`~qiskit_fermions.operators.FermionOperator.from_1body_tril_spin_sym` and
:meth:`~qiskit_fermions.operators.FermionOperator.from_2body_tril_spin_sym` expect the integrals
in packed (lower-triangular) `chemist` ordering, which is exactly what PySCF produces.

.. tab-set::

   .. tab-item:: Python
      :sync: python

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

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // Given the packed integrals (obtained as above), the C API exposes matching
         // electronic-integral constructors:
         //
         //   QfFermionOperator *h1 = qf_ferm_op_from_1body_tril_spin_sym(h1e_tril, norb);
         //   QfFermionOperator *h2 = qf_ferm_op_from_2body_tril_spin_sym(h2e_tril, norb);

3. Build the LUCJ circuit
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.UCJ` gate assembles the ansatz directly from the coupled-cluster amplitudes. Its
:meth:`~qiskit_fermions.circuit.library.UCJ.from_t_amplitudes` constructor performs a *double
factorization* of the :math:`t_2` amplitudes (via
:func:`~qiskit_fermions.linalg.double_factorized_t2`) to obtain the per-layer diagonal Coulomb
matrices and orbital rotations, and derives an optional final orbital rotation from the
:math:`t_1` amplitudes.

The number of ansatz repetitions :math:`L` equals the number of terms in the double
factorization. Truncating it with the ``n_reps`` argument trades some accuracy for a shallower
circuit; here we keep the two largest terms, which recovers most of the correlation energy while
halving the number of layers.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> from qiskit_fermions.circuit import FermionicCircuit
         >>> from qiskit_fermions.circuit.library import UCJ
         >>>
         >>> ansatz = UCJ.from_t_amplitudes(nelec, t2, t1=t1, n_reps=2)
         >>>
         >>> circuit = FermionicCircuit(2 * norb)
         >>> circuit.append(ansatz, circuit.modes)

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // The C API for FermionicCircuit is not available yet.

Decomposing the gate reveals its anatomy: an :class:`.InitializeModes` gate prepares the
Hartree-Fock reference determinant, and each ansatz layer contributes an :class:`.OrbitalRotation`
:math:`\mathcal{U}_k^\dagger`, then :math:`e^{i\mathcal{J}_k}` (an :class:`.Evolution` of the
diagonal Coulomb operator :math:`\mathcal{J}_k`), then :math:`\mathcal{U}_k`, with a final
:class:`.OrbitalRotation` at the end. The orbital rotations act per spin sector, so each is placed
on the alpha modes ``0..norb`` and the beta modes ``norb..2*norb`` independently.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because every gate in the circuit implements ffsim's :class:`ffsim.SupportsApplyUnitary` protocol,
the whole :class:`.FermionicCircuit` can be applied to a fixed particle-number state vector with
:func:`ffsim.apply_unitary`, starting from the Hartree-Fock reference. The
:class:`.FermionOperator` likewise implements ffsim's :class:`ffsim.SupportsLinearOperator`
protocol, so we can obtain a SciPy :class:`~scipy.sparse.linalg.LinearOperator` for it via
:func:`ffsim.linear_operator` and evaluate the ansatz energy as the expectation value of the
molecular Hamiltonian.

.. tab-set::

   .. tab-item:: Python
      :sync: python

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

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // Circuit simulation is not available via the C API.

The LUCJ energy improves substantially on the Hartree-Fock reference and approaches the CCSD
energy it was initialized from -- the small remaining gap is the price of truncating the ansatz to
two repetitions:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> print(f"Hartree-Fock: {scf.e_tot:.8f} Hartree")
   Hartree-Fock: -1.12675532 Hartree
   >>> print(f"CCSD:         {ccsd.e_tot:.8f} Hartree")
   CCSD:         -1.15167268 Hartree

.. note::
   The :class:`.UCJ` gate is initialized here from an *exact* double factorization of the
   :math:`t_2` amplitudes. ffsim additionally offers an optimized ("compressed") double
   factorization (its ``from_t_amplitudes(..., optimize=True)``), which has no equivalent in this
   package. To use it, build an ``ffsim`` UCJ operator with ``optimize=True`` and pass its
   ``diag_coulomb_mats`` / ``orbital_rotations`` / ``final_orbital_rotation`` into :class:`.UCJ`
   directly.

Next steps
^^^^^^^^^^

- Learn how the individual gates work in the :mod:`qiskit_fermions.circuit.library`
  documentation and the :ref:`fermionic circuit guide <fermionic_circuit_explanation>`.
- Explore the :ref:`operators explanation guide <operators_explanation>` to understand how to
  construct fermionic Hamiltonians such as the diagonal Coulomb operator used above.
- See how a fermionic circuit is mapped to qubits in the
  :ref:`transpilation guide <transpilation_explanation>`.

.. _LUCJ: https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc02516k
