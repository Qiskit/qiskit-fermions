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
shows how to drive that classical optimization -- the VQE **outer loop** -- around a fermionic
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
calculation, the Hamiltonian construction, and the ansatz gate itself in more detail -- this guide
focuses on the optimization loop around it instead. We reuse the molecule's CCSD :math:`t_1`/
:math:`t_2` amplitudes as the *initial point* for the optimization, rather than for a fixed ansatz.

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
parametrizing the *unconstrained* generators and mapping them onto valid tensors -- an orbital
rotation :math:`U = \exp(A)` for a complex anti-Hermitian generator :math:`A`, and a diagonal
Coulomb matrix written directly by its upper triangle and diagonal, then symmetrized.

.. tip::
   The rest of this guide only ever calls ``num_parameters``/``from_parameters``/``to_parameters``
   on the ansatz -- never anything :class:`.UCJ`-specific. Any ansatz gate exposing that same
   three-method interface can be dropped in here unchanged.

We fix a single repetition (``n_reps=1``) here to keep the optimization fast for this guide, and
keep the final orbital rotation from the :math:`t_1` amplitudes out of ``theta`` (see step 3) --
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
energy directly, ``params_to_vec`` stops one step earlier and returns the ansatz *state vector* --
this is the interface both optimizers used below need, and the energy is trivially recovered from
it via the Hamiltonian's :func:`ffsim.linear_operator`.

The final orbital rotation is initialized from the :math:`t_1` amplitudes and held fixed throughout:
since that rotation already captures the singles, the optimization can focus on the
:math:`t_2`-derived repetition tensors, and ``theta`` stays :math:`\mathcal{O}(N^2)` parameters
shorter. This is a choice, not a requirement -- freezing it does restrict the variational manifold,
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

4. Run the outer optimization loop with a generic optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting the optimizer from :math:`\boldsymbol{\theta} = \mathbf{0}` (the identity rotation and a
zero diagonal Coulomb matrix, i.e. the Hartree-Fock reference itself) lands on a nearby local
minimum barely below Hartree-Fock: with only one repetition, the ansatz is not expressive enough
near that point for a gradient-based search to find its way out unassisted. A classically computed
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
   gradients, the exact optimization trajectory -- in particular the number of iterations needed --
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

Even with a generous iteration budget, each ``energy`` evaluation only reveals a single scalar --
a wasteful use of the full state vector ``params_to_vec`` already computed internally.

5. Run the outer optimization loop with ffsim's linear method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`ffsim`_ ships :func:`ffsim.optimize.minimize_linear_method`, an optimizer purpose-built for
wavefunction ansatzes: rather than a scalar ``energy``, it takes ``params_to_vec`` directly (the
function returning the *state*, as built in step 3 above) together with the Hamiltonian
:class:`~scipy.sparse.linalg.LinearOperator`, and uses the extra structure this exposes --
gradients and an approximate Hessian of the state with respect to :math:`\boldsymbol{\theta}` --
to take much better-informed steps than a generic finite-difference method can. See `ffsim's own
how-to guide on simulating VQE
<https://qiskit-community.github.io/ffsim/how-to-guides/simulate-vqe.html>`_ for the method's
background and a walkthrough using ffsim's own ansatz classes; here, the same optimizer is applied
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

The exact iteration counts of both optimizers vary between runs and machines, so we don't print
them here -- but with the same warm start, the linear method consistently needs only a small
fraction of L-BFGS-B's iterations to reach the same energy, since it exploits the extra structure
of ``params_to_vec`` that a generic finite-difference method cannot.

.. skip: end

Next steps
^^^^^^^^^^

- See the :ref:`LUCJ guide <lucj_getting_started>` for how :class:`.UCJ` is normally constructed
  from coupled-cluster amplitudes, and how to transpile it onto qubits for hardware execution --
  the same transpilation applies to ``UCJ.from_parameters(lm_result.x, ...)`` here, built from the
  converged parameter vector from step 5.
- Read the :ref:`ffsim backend guide <ffsim_backend_explanation>` for why :func:`ffsim.apply_unitary`
  and :func:`ffsim.linear_operator` work natively on :class:`.FermionicCircuit` and
  :class:`.FermionOperator`.
- See `ffsim's how-to guide on simulating VQE
  <https://qiskit-community.github.io/ffsim/how-to-guides/simulate-vqe.html>`_ for a walkthrough
  of :func:`~ffsim.optimize.minimize_linear_method` using ffsim's own ansatz classes, and for tuning
  the linear method's other hyperparameters (``regularization``, ``variation``, and more).

.. _ffsim: https://qiskit-community.github.io/ffsim/
