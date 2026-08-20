.. _vqe_outer_loop_how_to:

Running the outer VQE parameter optimization loop
==================================================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available through the C API in a future
   release.

The variational quantum eigensolver (VQE) minimizes the expectation value of a Hamiltonian over a
parametrized family of states,

.. math::

   E(\boldsymbol{\theta}) = \langle \Psi(\boldsymbol{\theta}) \rvert H \lvert
   \Psi(\boldsymbol{\theta}) \rangle,

using a classical optimizer to drive :math:`\boldsymbol{\theta}` toward the minimum. This guide
shows how to drive that classical optimization (the VQE outer loop) around a fermionic
ansatz gate, using :class:`.UCJ` as a concrete example.

.. note::
   Fermionic ansatz gates such as :class:`.UCJ` are built from concrete numeric tensors
   (``diag_coulomb_mats``, ``orbital_rotations``), not from :class:`~qiskit.circuit.Parameter`
   objects: their synthesis (see :class:`.GivensDecompositionOrbitalRotationSynthesis`) performs a
   numeric Givens decomposition of the rotation matrix, which has no symbolic equivalent. So rather
   than binding parameters into one fixed circuit, this guide drives an outer loop: a classical
   optimizer proposes a flat real vector :math:`\boldsymbol{\theta}`, which
   :meth:`.UCJ.from_parameters` unpacks into a fresh ansatz gate, evaluated once per optimizer step.
   This mirrors how `ffsim`_'s own VQE examples treat its equivalent UCJ operators.

.. invisible-code-block: python

   >>> from qiskit_fermions.utils.optionals import HAS_FFSIM

.. skip: start if(not HAS_FFSIM)

1. Construct the ansatz
^^^^^^^^^^^^^^^^^^^^^^^

As a concrete example, we use the :class:`.UCJ` ansatz for a hydrogen molecule in the ``6-31g``
basis, following the :ref:`LUCJ guide <lucj_getting_started>`, which covers the classical
calculation, the Hamiltonian construction, and the ansatz gate itself in more detail; this guide
focuses on the optimization loop around it instead. We reuse the molecule's coupled-cluster
singles and doubles (CCSD) :math:`t_1`/
:math:`t_2` amplitudes as the initial point for the optimization, rather than for a fixed ansatz.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import pyscf
   >>> import pyscf.cc
   >>>
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
   >>> ccsd = pyscf.cc.CCSD(scf).run()
   >>> t1, t2 = ccsd.t1, ccsd.t2

The molecular Hamiltonian is built the same way as in the LUCJ guide:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from pyscf import ao2mo, lib
   >>>
   >>> from qiskit_fermions.operators import FermionOperator
   >>>
   >>> h1e = mo_coeff.T @ scf.get_hcore() @ mo_coeff
   >>> h2e = ao2mo.kernel(mol, mo_coeff)
   >>> ecore = mol.energy_nuc()
   >>>
   >>> h1e_tril = lib.pack_tril(h1e)
   >>> h2e_tril = lib.pack_tril(h2e)
   >>>
   >>> hamiltonian = ecore * FermionOperator.one()
   >>> hamiltonian += FermionOperator.from_1body_tril_spin_sym(h1e_tril, norb)
   >>> hamiltonian += FermionOperator.from_2body_tril_spin_sym(h2e_tril, norb)

2. Choose an unconstrained parametrization of the ansatz tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optimizer needs a flat real vector, but the ansatz tensors are constrained: each orbital
rotation must be unitary, and each diagonal Coulomb matrix must be real symmetric.
:meth:`.UCJ.to_parameters` and :meth:`.UCJ.from_parameters` handle this conversion natively,
parametrizing the *unconstrained* generators and mapping them onto valid tensors: an orbital
rotation :math:`U = \exp(A)` for a complex anti-Hermitian generator :math:`A`, and a diagonal
Coulomb matrix written directly by its upper triangle and diagonal, then symmetrized.

.. tip::
   The rest of this guide only ever calls ``num_parameters``/``from_parameters``/``to_parameters``
   on the ansatz, never anything :class:`.UCJ`-specific. Any ansatz gate exposing that same
   three-method interface can be dropped in here unchanged; :ref:`step 6 <vqe_outer_loop_ucc>` does
   that with :class:`.UCC`.

We fix a single repetition (``n_reps=1``) here to keep the optimization fast for this guide, and
keep the final orbital rotation from the :math:`t_1` amplitudes out of ``theta`` (see step 3);
:meth:`.UCJ.num_parameters` reports how many parameters this leaves.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import numpy as np
   >>>
   >>> from qiskit_fermions.circuit.library import UCJ
   >>>
   >>> n_reps = 1
   >>> num_params = UCJ.num_parameters(norb, "balanced", n_reps)

3. Build the ansatz and evaluate the state from a flat parameter vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given ``theta``, :meth:`.UCJ.from_parameters` builds the gate directly. Rather than returning an
energy directly, ``params_to_vec`` stops one step earlier and returns the ansatz state vector:
this is the interface both optimizers used below need, and the energy is trivially recovered from
it via the Hamiltonian's :func:`ffsim.linear_operator`.

The final orbital rotation is initialized from the :math:`t_1` amplitudes and held fixed throughout:
since that rotation already captures the singles, the optimization can focus on the
:math:`t_2`-derived repetition tensors, and ``theta`` stays :math:`\mathcal{O}(N^2)` parameters
shorter. This is a choice, not a requirement: freezing it does restrict the variational manifold,
so for systems with significant singles character you may well want it optimized too. Passing
``with_final_orbital_rotation=True`` to both :meth:`.UCJ.num_parameters` and
:meth:`.UCJ.from_parameters` folds it into ``theta``, which then also makes the two explicit
``final_orbital_rotation`` assignments below unnecessary.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import ffsim
   >>>
   >>> from qiskit_fermions.circuit import FermionicCircuit
   >>> from qiskit_fermions.circuit.library import InitializeModes, OrbitalRotation, UCJ
   >>>
   >>> final_orbital_rotation = OrbitalRotation.from_t1_amplitudes(t1).rotation_unitary
   >>>
   >>> reference = ffsim.hartree_fock_state(norb, nelec)
   >>> linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
   >>>
   >>> def params_to_vec(theta):
   ...     """Builds a fresh UCJ ansatz from theta and returns its state vector."""
   ...     ansatz = UCJ.from_parameters(theta, norb, "balanced", n_reps)
   ...     # reattached rather than read from theta, so it stays fixed; to optimize it instead, pass
   ...     # with_final_orbital_rotation=True above (and to num_parameters) and drop this line
   ...     ansatz.final_orbital_rotation = final_orbital_rotation
   ...
   ...     circuit = FermionicCircuit(2 * norb)
   ...     circuit.append(InitializeModes.from_hartree_fock(norb, nelec), circuit.modes)
   ...     circuit.append(ansatz, circuit.modes)
   ...
   ...     return ffsim.apply_unitary(reference, circuit, norb=norb, nelec=nelec)
   >>>
   >>> def energy(theta):
   ...     """Returns the Hamiltonian expectation value of the ansatz state for theta."""
   ...     state = params_to_vec(theta)
   ...     return np.vdot(state, linop @ state).real

Each call to ``params_to_vec`` builds an entirely new :class:`.UCJ` gate: there is no persistent
circuit carrying bound parameters, only the mapping from ``theta`` to tensors to a gate to a state.

4. Run the outer loop with a generic optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting the optimizer from :math:`\boldsymbol{\theta} = \mathbf{0}` (the identity rotation and a
zero diagonal Coulomb matrix, that is, the Hartree-Fock reference itself) lands on a nearby local
minimum barely below Hartree-Fock. With only one repetition, the ansatz is not expressive enough
near that point for a gradient-based search to escape it unassisted. A classically computed
starting point is a much better choice, so we reuse the CCSD-initialized ansatz from step 1: calling
:meth:`~.UCJ.to_parameters` on it (with its final rotation cleared, since ``theta`` excludes it)
gives ``theta0``, which is then handed to :func:`scipy.optimize.minimize` together with ``energy``.
Any ``scipy`` gradient-free or finite-difference method works here since ``energy`` returns a plain
float; no gradient of the fermionic ansatz is implemented.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from scipy.optimize import minimize
   >>>
   >>> ccsd_ansatz = UCJ.from_t_amplitudes(nelec, t2, t1=t1, n_reps=1)
   >>> # cleared so that to_parameters emits theta's layout; keep it if you opted to optimize it
   >>> ccsd_ansatz.final_orbital_rotation = None
   >>> theta0 = ccsd_ansatz.to_parameters()
   >>>
   >>> result = minimize(
   ...     energy, theta0, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8}
   ... )

.. note::
   Because ``energy`` relies on multi-threaded linear algebra (inside :func:`ffsim.apply_unitary`
   and :func:`ffsim.linear_operator`), and ``minimize`` here uses numeric finite-difference
   gradients, the optimization trajectory (in particular the number of iterations needed)
   can vary slightly between runs, machines, and BLAS backends. We give the optimizer a generous
   iteration budget and check convergence with a tolerance rather than pinning down an exact
   iteration count.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> print(f"Hartree-Fock: {scf.e_tot:.5f} Hartree")
   Hartree-Fock: -1.12676 Hartree
   >>> print(f"CCSD:         {ccsd.e_tot:.5f} Hartree")
   CCSD:         -1.15167 Hartree
   >>> print(f"L-BFGS-B:     {result.fun:.5f} Hartree")
   L-BFGS-B:     -1.15167 Hartree
   >>> bool(abs(result.fun - ccsd.e_tot) < 1e-4)
   True

Even with a generous iteration budget, each ``energy`` evaluation only reveals a single scalar,
a wasteful use of the full state vector ``params_to_vec`` already computed internally.

5. Run the outer loop with ffsim's linear method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`ffsim`_ ships :func:`ffsim.optimize.minimize_linear_method`, an optimizer purpose-built for
wavefunction ansatzes: rather than a scalar ``energy``, it takes ``params_to_vec`` directly (the
function returning the state, as built in step 3 above) together with the Hamiltonian
:class:`~scipy.sparse.linalg.LinearOperator`, and uses the extra structure this exposes
(gradients and an approximate Hessian of the state with respect to :math:`\boldsymbol{\theta}`)
to take much better-informed steps than a generic finite-difference method can. See `ffsim's own
how-to guide on simulating VQE
<https://qiskit-community.github.io/ffsim/how-to-guides/simulate-vqe.html>`_ for the method's
background and a walkthrough using ffsim's own ansatz classes; the same optimizer is applied
to the :class:`.UCJ` gate and ``params_to_vec`` built above, unchanged.

.. note::
   By default, each step also runs an inner search over the ``regularization``/``variation``
   hyperparameters, which can itself become numerically sensitive once the ansatz is already very
   close to the optimum, occasionally causing a step to stall. Since a good fixed ``regularization``
   and ``variation`` are already known to work well starting this close to the CCSD solution, we
   disable that inner search here for a reliably reproducible trajectory.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> lm_result = ffsim.optimize.minimize_linear_method(
   ...     params_to_vec,
   ...     linop,
   ...     x0=theta0,
   ...     maxiter=50,
   ...     optimize_regularization=False,
   ...     optimize_variation=False,
   ... )
   >>>
   >>> print(f"CCSD:          {ccsd.e_tot:.5f} Hartree")
   CCSD:          -1.15167 Hartree
   >>> print(f"linear method: {lm_result.fun:.5f} Hartree")
   linear method: -1.15167 Hartree

The iteration counts of both optimizers vary between runs and machines, so we don't print
them here; but with the same warm start, the linear method consistently needs only a small
fraction of L-BFGS-B's iterations to reach the same energy, since it exploits the extra structure
of ``params_to_vec`` that a generic finite-difference method cannot.

.. _vqe_outer_loop_ucc:

6. Swap in a different ansatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nothing above is specific to :class:`.UCJ`. Because the loop only ever calls
``num_parameters``/``from_parameters``/``to_parameters``, another ansatz exposing that interface
drops straight in. :class:`.UCC` implements the unitary coupled-cluster ansatz
:math:`e^{T - T^\dagger}` and exposes those three methods, so the only changes are the
class name and its shape arguments: :class:`.UCC` is sized by the occupied/virtual split
(``norb``, ``nocc``) rather than by a repetition count.

.. note::
   The variant names differ between the two classes but describe the same spin choice here:
   :class:`.UCJ`'s ``"balanced"`` and :class:`.UCC`'s ``"restricted"`` both mean one spatial
   parametrization shared by both spin sectors. Each class follows the naming of the ffsim operators
   it mirrors (:external:class:`~ffsim.UCJOpSpinBalanced` versus
   :external:class:`~ffsim.UCCSDOpRestrictedReal`). Likewise :class:`.UCJ`'s ``"unbalanced"`` and
   :class:`.UCC`'s ``"unrestricted"`` are the independent-per-spin variants, and both classes call
   the single-register variant ``"spinless"``.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.circuit.library import UCC
   >>>
   >>> nocc = nelec[0]
   >>>
   >>> def ucc_params_to_vec(theta):
   ...     """Builds a fresh UCC ansatz from theta and returns its state vector."""
   ...     ansatz = UCC.from_parameters(theta, norb, nocc, "restricted")
   ...
   ...     circuit = FermionicCircuit(2 * norb)
   ...     circuit.append(InitializeModes.from_hartree_fock(norb, nelec), circuit.modes)
   ...     circuit.append(ansatz, circuit.modes)
   ...
   ...     return ffsim.apply_unitary(reference, circuit, norb=norb, nelec=nelec)

Note the absent ``final_orbital_rotation``: :class:`.UCC` carries none, because its :math:`t_1`
amplitudes already *are* its single excitations. For :class:`.UCJ` that trailing rotation is where
the singles live (it only factorizes :math:`t_2`), which is why steps 3 and 4 had to hold it
fixed outside ``theta``. Every amplitude is a parameter, so :meth:`.UCC.to_parameters` gives
the warm start directly, and the whole question of freezing a rotation does not arise. Should you
want one anyway, to widen the manifold beyond what the singles already span, append an
:class:`.OrbitalRotation` to the circuit yourself; unlike UCJ's flag, it is then yours to either
keep fixed or fold into ``theta`` by hand.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> ucc_theta0 = UCC.from_t_amplitudes(t2, t1=t1, variant="restricted").to_parameters()
   >>>
   >>> ucc_result = ffsim.optimize.minimize_linear_method(
   ...     ucc_params_to_vec,
   ...     linop,
   ...     x0=ucc_theta0,
   ...     maxiter=50,
   ...     optimize_regularization=False,
   ...     optimize_variation=False,
   ... )
   >>>
   >>> print(f"CCSD:      {ccsd.e_tot:.5f} Hartree")
   CCSD:      -1.15167 Hartree
   >>> print(f"UCC:       {ucc_result.fun:.5f} Hartree")
   UCC:       -1.15167 Hartree

Both ansatzes reach the CCSD energy on this molecule. Their parameter counts differ, but be careful
reading anything general into that at this size; the two scale quite differently. :class:`.UCJ`'s
count is :math:`O(n_\text{orb}^2)` per repetition, while :class:`.UCC`'s doubles are
:math:`O(n_\text{occ}^2 n_\text{vrt}^2)`, so UCC starts smaller on tiny systems and ends up much
larger. At half filling the crossover is already around eight orbitals:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> for n in (4, 8, 20):
   ...     ucj = UCJ.num_parameters(n, "balanced", 1)
   ...     ucc = UCC.num_parameters(n, n // 2, "restricted")
   ...     print(f"norb={n:<3} UCJ (n_reps=1): {ucj:<5} UCC: {ucc}")
   norb=4   UCJ (n_reps=1): 36    UCC: 14
   norb=8   UCJ (n_reps=1): 136   UCC: 152
   norb=20  UCJ (n_reps=1): 820   UCC: 5150

This is a trade-off, not a ranking, and the comparison above is deliberately generous to UCJ by
holding ``n_reps=1``: repetitions are what buy UCJ its expressivity, and its count grows linearly in
them. What matters for this guide is that switching between the two costs three lines.

.. note::
   The two ansatzes also differ in how faithfully their circuits reproduce the gate. :class:`.UCJ`
   synthesizes exactly, whereas :class:`.UCC`'s excitation terms do not commute, so its circuit
   :meth:`~qiskit.circuit.Gate.definition` is a first-order product formula. The simulation path used
   above applies the exponential exactly, so the optimization here is unaffected; but a circuit
   transpiled for hardware carries a Trotter error, tightened with a higher-order product formula.

.. skip: end

Next steps
^^^^^^^^^^

- See the :ref:`LUCJ guide <lucj_getting_started>` for how :class:`.UCJ` is normally constructed
  from coupled-cluster amplitudes, and how to transpile it onto qubits for hardware execution;
  the same transpilation applies to ``UCJ.from_parameters(lm_result.x, ...)`` here, built from the
  converged parameter vector from step 5.
- See :class:`.UCC` for the unitary coupled-cluster ansatz swapped in at step 6, including its
  ``"unrestricted"`` and ``"spinless"`` variants and the opt-in ``antisymmetric`` parameterization
  of the :math:`t_2` amplitudes.
- Read the :ref:`ffsim backend guide <ffsim_backend_explanation>` for why :func:`ffsim.apply_unitary`
  and :func:`ffsim.linear_operator` work natively on :class:`.FermionicCircuit` and
  :class:`.FermionOperator`.
- See `ffsim's how-to guide on simulating VQE
  <https://qiskit-community.github.io/ffsim/how-to-guides/simulate-vqe.html>`_ for a walkthrough
  of :func:`~ffsim.optimize.minimize_linear_method` using ffsim's own ansatz classes, and for tuning
  the linear method's other hyperparameters (``regularization``, ``variation``, and more).

.. _ffsim: https://qiskit-community.github.io/ffsim/
