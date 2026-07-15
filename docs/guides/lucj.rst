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

The ansatz parameters are derived entirely with this package's own
:mod:`~qiskit_fermions.linalg` module, so building the circuit needs only ``qiskit-fermions``
together with NumPy and SciPy. This guide additionally uses `PySCF <https://pyscf.org/>`_ to
run the underlying quantum chemistry and the optional ``ffsim`` dependency (managed by
:data:`.HAS_FFSIM`) to package the integrals and simulate the resulting circuit.

.. invisible-code-block: python

   >>> from qiskit_fermions.utils.optionals import HAS_FFSIM
   >>> from qiskit.utils import LazyImportTester
   >>> HAS_PYSCF = LazyImportTester("pyscf", name="PySCF")

.. skip: start if(not HAS_FFSIM or not HAS_PYSCF)

1. Obtain the cluster amplitudes from a classical calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The (L)UCJ ansatz can be initialized from the amplitudes of a coupled-cluster singles and
doubles (CCSD) calculation. Here we run restricted Hartree-Fock followed by CCSD for a
hydrogen molecule in the ``6-31g`` basis. We use ``pyscf`` for the quantum chemistry and
``ffsim`` only to package the active-space integrals (and, later, to simulate the ansatz).

.. code-block:: python

   >>> import pyscf
   >>> import pyscf.cc
   >>> import ffsim
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
   >>> # extract the active-space definition and run CCSD for the t-amplitudes
   >>> mol_data = ffsim.MolecularData.from_scf(scf)
   >>> norb, nelec = mol_data.norb, mol_data.nelec
   >>> ccsd = pyscf.cc.CCSD(scf).run()
   >>> t1, t2 = ccsd.t1, ccsd.t2

2. Factorize the amplitudes into ansatz layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The connection between the cluster amplitudes and the UCJ ansatz runs through a *double
factorization* of the :math:`t_2` amplitudes. Each term of the factorization is a
:math:`(Z, U)` pair -- a real symmetric diagonal Coulomb matrix :math:`Z` and a unitary
orbital rotation :math:`U` -- and together the terms define the ansatz layers
:math:`\mathcal{U}_k e^{i\mathcal{J}_k} \mathcal{U}_k^\dagger`. We use
:func:`~qiskit_fermions.linalg.double_factorized_t2` from this package's linear-algebra module
to perform the factorization directly on the CCSD amplitudes; no external ansatz object is
required.

.. code-block:: python

   >>> from qiskit_fermions.linalg import double_factorized_t2
   >>>
   >>> # each term is a (diagonal Coulomb matrix Z, orbital rotation U) pair defining one layer
   >>> terms = double_factorized_t2(t2.astype(complex), tol=1e-8)

The :math:`t_1` amplitudes contribute an optional final orbital rotation, constructed as
:math:`\exp(t_1 - t_1^\dagger)` after embedding the amplitudes into an anti-Hermitian
generator over all orbitals:

.. code-block:: python

   >>> import numpy as np
   >>> import scipy.linalg
   >>>
   >>> def final_rotation_from_t1(t1):
   ...     """Build the orbital rotation exp(t1 - t1^dagger) from the t1 amplitudes."""
   ...     nocc, nvrt = t1.shape
   ...     generator = np.zeros((nocc + nvrt, nocc + nvrt), dtype=complex)
   ...     generator[:nocc, nocc:] = -t1.conj()
   ...     generator[nocc:, :nocc] = t1.T
   ...     return scipy.linalg.expm(generator)
   >>>
   >>> final_orbital_rotation = final_rotation_from_t1(t1)

3. Translate the layers into fermionic operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each diagonal Coulomb layer is a :class:`.FermionOperator`. We use the block-spin mode
convention that the fermionic circuit shares with the simulation backend: mode :math:`p` is
the spin-up (alpha) orbital :math:`p`, and mode :math:`\text{norb} + p` is the spin-down
(beta) orbital :math:`p`. A number operator is then :math:`n_m = a^\dagger_m a_m`, so each
product :math:`n_{i\sigma} n_{j\tau}` is a single four-operator term.

For the spin-balanced ansatz, a single matrix :math:`Z` describes each layer: the alpha-alpha
and beta-beta blocks both equal :math:`Z`, and the alpha-beta and beta-alpha blocks equal
:math:`Z` and its transpose (here :math:`Z` is symmetric, so they coincide).

.. code-block:: python

   >>> from qiskit_fermions.operators import FermionOperator, cre, ann
   >>>
   >>> def diag_coulomb_operator(mat, norb):
   ...     """Build J = 1/2 sum_{ij,st} Z_{ij} n_{i,s} n_{j,t} as a FermionOperator."""
   ...     blocks = {(0, 0): mat, (0, 1): mat, (1, 0): mat.T, (1, 1): mat}
   ...     terms = {}
   ...     for (sigma, tau), block in blocks.items():
   ...         for i in range(norb):
   ...             for j in range(norb):
   ...                 coeff = 0.5 * block[i, j]
   ...                 if coeff == 0.0:
   ...                     continue
   ...                 mode_i, mode_j = sigma * norb + i, tau * norb + j
   ...                 term = (cre(mode_i), ann(mode_i), cre(mode_j), ann(mode_j))
   ...                 terms[term] = terms.get(term, 0.0) + coeff
   ...     return FermionOperator.from_dict(terms)

The orbital rotations act on both spin sectors with the same ``norb x norb`` matrix. A
:class:`.OrbitalRotation` gate acts on exactly the modes it is placed on, so instead of
embedding the rotation into a larger block-diagonal matrix we simply append it twice: once to
the alpha modes ``0..norb`` and once to the beta modes ``norb..2*norb``. Because the two spin
sectors are disjoint, the sectors never mix and the placements commute.

.. code-block:: python

   >>> from qiskit_fermions.circuit.library import OrbitalRotation
   >>>
   >>> def add_orbital_rotation(circuit, rotation, norb):
   ...     """Append a per-spin orbital rotation to the alpha and beta halves of the register."""
   ...     circuit.append(OrbitalRotation(rotation), circuit.modes[:norb])
   ...     circuit.append(OrbitalRotation(rotation), circuit.modes[norb:])

4. Assemble the LUCJ circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now build the :class:`.FermionicCircuit`. It opens with an :class:`.InitializeModes` gate
that prepares the Hartree-Fock reference determinant (the first ``n_alpha`` alpha modes and
first ``n_beta`` beta modes occupied). Each factorization term contributes the sequence
:math:`\mathcal{U}_k^\dagger`, then :math:`e^{i\mathcal{J}_k}` (an :class:`.Evolution` with
``time=-1`` so that :math:`e^{-i(-1)\mathcal{J}_k} = e^{i\mathcal{J}_k}`), then
:math:`\mathcal{U}_k`. Finally we append the :math:`t_1`-derived orbital rotation.

.. code-block:: python

   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import Evolution, InitializeModes
   >>>
   >>> n_alpha, n_beta = nelec
   >>> occupation = [False] * (2 * norb)
   >>> for i in range(n_alpha):
   ...     occupation[i] = True
   >>> for i in range(n_beta):
   ...     occupation[norb + i] = True
   >>>
   >>> circuit = FermionicCircuit(2 * norb)
   >>> circuit.append(InitializeModes(occupation), circuit.modes)
   >>>
   >>> for diag_coulomb_mat, orbital_rotation in terms:
   ...     diag_coulomb = diag_coulomb_operator(diag_coulomb_mat, norb)
   ...     add_orbital_rotation(circuit, orbital_rotation.conj().T, norb)
   ...     circuit.append(Evolution(2 * norb, diag_coulomb, time=-1.0), circuit.modes)
   ...     add_orbital_rotation(circuit, orbital_rotation, norb)
   >>>
   >>> add_orbital_rotation(circuit, final_orbital_rotation, norb)

5. Simulate the ansatz and evaluate its energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because every gate in the circuit implements ``ffsim``'s ``SupportsApplyUnitary`` protocol,
the whole :class:`.FermionicCircuit` can be applied directly to a fixed particle-number state
vector. The :class:`.InitializeModes` gate seeds the reference determinant, so we can apply
the circuit starting from no incoming state (``None``). We then compute the ansatz energy as
the expectation value of the molecular Hamiltonian.

.. code-block:: python

   >>> state = circuit._apply_unitary_(None, norb, nelec, copy=True)
   >>>
   >>> hamiltonian = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)
   >>> energy = np.vdot(state, hamiltonian @ state).real
   >>>
   >>> # the LUCJ energy improves on the Hartree-Fock reference
   >>> bool(energy < scf.e_tot)
   True

.. skip: end

.. note::
   Deriving the ansatz layers from the CCSD amplitudes uses only the core
   :mod:`qiskit_fermions` API (:func:`~qiskit_fermions.linalg.double_factorized_t2`) plus NumPy
   and SciPy. Running the classical chemistry requires PySCF, and packaging the integrals and
   simulating the circuit require the optional dependency managed by :data:`.HAS_FFSIM`.

Next steps
^^^^^^^^^^

- Learn how the individual gates work in the :mod:`qiskit_fermions.circuit.library`
  documentation and the :ref:`fermionic circuit guide <fermionic_circuit_explanation>`.
- Explore the :ref:`operators explanation guide <operators_explanation>` to understand how to
  construct fermionic Hamiltonians such as the diagonal Coulomb operator used above.
- See how a fermionic circuit is mapped to qubits in the
  :ref:`transpilation guide <transpilation_explanation>`.

.. _LUCJ: https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc02516k
