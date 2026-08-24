.. _sqdrift_getting_started:

Generate SqDRIFT circuits
=========================

With sample-based quantum diagonalization (`SQD`_), you must choose an ansatz from which to sample
bitstrings.
The `SqDRIFT`_ variant uses an ensemble of time evolution circuits
constructed directly from the target Hamiltonian instead.
This is achieved by subsampling smaller time evolution operators from the
Hamiltonian based on its coefficients, which is known as the `qDRIFT`_
Trotterization method.

This getting started guide shows how to generate an ensemble of such randomized
circuits.

1. Hamiltonian setup
^^^^^^^^^^^^^^^^^^^^

For the purposes of this guide, load the electronic structure Hamiltonian
of N2 from an FCIDUMP file. There are other means of
constructing the :class:`.FermionOperator`. Be sure to consult its
documentation, as well as the :mod:`qiskit_fermions.operators.library`.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.operators.library import FCIDump
       >>> from qiskit_fermions.operators import FermionOperator
       >>>
       >>> fcidump = FCIDump.from_file("docs/guides/n2.fcidump")
       >>> num_modes = 2 * fcidump.norb
       >>> hamil = FermionOperator.from_fcidump(fcidump)

    .. code-block:: c

       #include <qiskit_fermions.h>

       QfFCIDump* fcidump = qf_fcidump_from_file("docs/guides/n2.fcidump");
       QfFermionOperator* hamil = qf_ferm_op_from_fcidump(fcidump);
       uint32_t num_modes = 2 * qf_fcidump_norb(fcidump);

2. Group Hamiltonian terms
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the many symmetries that are present in the electronic
structure Hamiltonian by grouping related terms that have identical coefficients.
This action changes the operator coefficient distribution that the qDRIFT
protocol samples from, but it does not affect its convergence guarantees.
Crucially, grouping terms that are related by symmetry results in a favorable
cancellation of Pauli terms, resulting in an overall shorter circuit depth when
time evolving a state under their action.

The :mod:`qiskit_fermions.operators.terms.grouping` module provides convenience
functions for grouping an operator's terms. This is explained in more
detail in :ref:`this guide <grouping_explanation>`.

.. caution::
   The implementation of the :func:`~group_terms_by_electronic_structure`
   assumes the terms of the Hamiltonian to be normal-ordered!

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
       >>> from qiskit_fermions.operators.terms.ordering import canonical_order
       >>>
       >>> canon = canonical_order(hamil.normal_ordered().simplify(atol=1e-16))
       >>> exit_code = group_terms_by_electronic_structure(canon, num_modes, two_body_physicist_order=False)
       >>> assert exit_code is None
       >>> print(canon.groups)  # the groups attribute now contains some list of group indices
       [0, ...]

    .. code-block:: c

       QfFermionOperator* normal;
       QfExitCode exit = qf_ferm_op_group_terms_by_electronic_structure(normal, num_modes, false);
       QfFermionOperator* canon = qf_ferm_op_canonical_order(normal);

.. hint::

   The full electronic structure Hamiltonian contains certain terms whose
   inclusion in a time-evolution circuit has no impact on the perceived
   bitstrings and, thus, only results in an increased sampling overhead.
   Therefore, it is recommended that such terms be filtered from the
   Hamiltonian at this point, before constructing the :class:`.Evolution` gate
   in the next step.

   The terms that fit this description are those that are diagonal
   in the occupation-number basis, that is, the products of number operators
   (:math:`a^\dagger_i a_i`). This includes the constant energy offset, whose
   time evolution only introduces a global phase into the circuit, the
   individual number-operators whose time evolution amounts to single-qubit Z
   rotations, as well as higher-order products such as :math:`n_i n_j`. None
   of these impact the sampled bitstrings.

   The :func:`~qiskit_fermions.operators.terms.filtering.filter_diagonal_terms`
   function removes such terms from an operator in place:

   .. tab-set-code::

       .. code-block:: python

          >>> from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms
          >>>
          >>> filter_diagonal_terms(canon)

       .. code-block:: c

          qf_ferm_op_filter_diagonal_terms(canon);

   Filtering here, once, is considerably cheaper than filtering repeatedly.
   :class:`.QDriftTrotterization` runs once per transpiled circuit, so
   filtering the Hamiltonian initially, rather than on every call, avoids
   redoing work for every circuit generated from it.

3. Prepare the time evolution circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prepare the Hamiltonian's time evolution circuit and the
base circuit from which to draw samples. The
:mod:`qiskit_fermions.circuit.library` contains all the required
components to do so, in compliance with Qiskit conventions.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.circuit import FermionicCircuit
       >>> from qiskit_fermions.circuit.library import Evolution
       >>>
       >>> time = 1.0  # you can choose a desired scaling factor here
       >>> evo_gate = Evolution(num_modes, canon, time)
       >>>
       >>> circ = FermionicCircuit(num_modes)
       >>> circ.append(evo_gate, circ.modes)

    .. code-block:: c

       // WARNING: Qiskit's C API does not yet allow us to implement circuits
       // with custom gate definitions.

.. note::
   This example neither initializes the fermionic modes with particles,
   nor measures their final state.

4. Transpile the circuit with QDrift Trotterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`qiskit_fermions.transpiler` module integrates directly with Qiskit's
transpilation pipeline, allowing the :class:`.FermionicCircuit` constructed above
to be directly transpiled to a :external:class:`~qiskit.circuit.QuantumCircuit`.

Use the :func:`.jordan_wigner` fermion-to-qubit mapping to
convert the Hamiltonian expressed in terms of fermions to be expressed in Pauli strings instead. This
can be done directly as part of the transpilation process by using the
:class:`.EvolutionSynthesis` transpilation pass plugin. Use
:func:`.generate_preset_jw_pass_manager` to build
:class:`.FermionicStagedPassManager`, which ensures that the
Jordan-Wigner encoding is used consistently for all circuit instructions.

Crucially, add the :class:`.QDriftTrotterization` transpilation pass to the
``optimization`` stage of the transpilation pipeline. This ensures that the
circuit does not use the time evolution of the entire Hamiltonian, whose depth
would exceed the capabilities of currently available quantum computing hardware.

Instead, it subsamples a fixed number of ``groups`` of Hamiltonian terms for
each circuit, every time the circuit is transpiled. Through this, you can generate
multiple circuit randomizations as required by the `SqDRIFT`_ algorithm by
repeatedly running the transpilation pipeline.

This step also introduces the few parameters you can use to customize the
circuits to generate:

* The number of circuits to generate: ``num_sqdrift_randomizations``
* The length of each circuit in terms of excitation groups: ``num_groups``

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.transpiler import FermionicPassManager
       >>> from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
       >>> from qiskit_fermions.transpiler.passes import QDriftTrotterization
       >>>
       >>> num_groups = 10
       >>> qdrift = QDriftTrotterization(num_groups, rng=19)
       >>>
       >>> pm = generate_preset_jw_pass_manager()
       >>> pm.optimization = FermionicPassManager([qdrift])
       >>>
       >>> num_sqdrift_randomizations = 10
       >>> sqdrift_circuits = [
       ...     pm.run(circ) for _ in range(num_sqdrift_randomizations)
       ... ]

    .. code-block:: c

       // WARNING: Qiskit's C API does not yet allow us to implement circuits
       // with custom gate definitions, which we therefore also cannot transpile
       // via this API.

.. note::
   The preceding example fixes the ``seed`` for the random number
   generator used inside of the :class:`.QDriftTrotterization` transpilation
   pass.

5. Filter out trivial excitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond the diagonal terms filtered out in the previous step, a sampled
excitation can *still* fail to affect the sampled bitstrings. Whenever it acts
entirely within a set of modes whose occupation is already fixed (all occupied
or all unoccupied), it cannot move a particle from one to the other, so it
leaves the state (and thus the eventual measurement outcome) unchanged.
Setting ``filter_trivial=True`` on the :class:`.QDriftTrotterization` pass
rejects such terms as they are sampled and re-draws a replacement, so that
each of the ``num_groups`` slots of the resulting circuit contributes a
non-trivial excitation.

This filtering needs to know which modes start out occupied. It therefore
requires an :class:`.InitializeModes` gate preceding the :class:`.Evolution`
gate(s) in the circuit; :meth:`.InitializeModes.from_hartree_fock` is a
convenient way to construct one. Add one here for the N2 Hartree-Fock
reference (seven alpha and seven beta electrons in 14 spatial orbitals) and compare the
sampled excitations with and without ``filter_trivial=True``:

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.circuit.library import InitializeModes
       >>>
       >>> init = InitializeModes.from_hartree_fock(fcidump.norb, (7, 7))
       >>>
       >>> hf_circ = FermionicCircuit(num_modes)
       >>> hf_circ.append(init, hf_circ.modes)
       >>> hf_circ.append(Evolution(num_modes, canon, time), hf_circ.modes)
       >>>
       >>> num_groups = 5
       >>> qdrift_unfiltered = QDriftTrotterization(num_groups, rng=3480)
       >>> qdrift_trivial = QDriftTrotterization(
       ...     num_groups, filter_trivial=True, rng=3480
       ... )
       >>>
       >>> for instruction in FermionicPassManager(qdrift_unfiltered).run(hf_circ)._inner.data:
       ...     if instruction.operation.name == "Evolution":
       ...         print(sorted(instruction.operation.operator.get_support()))
       [2, 4]
       [41, 45, 52]
       [15, 45, 55]
       [41, 52, 53]
       [10, 16, 37, 38]
       >>>
       >>> for instruction in FermionicPassManager(qdrift_trivial).run(hf_circ)._inner.data:
       ...     if instruction.operation.name == "Evolution":
       ...         print(sorted(instruction.operation.operator.get_support()))
       [0, 1, 6, 7]
       [0, 1, 28, 29]
       [4, 13, 55]
       [13, 20, 40, 41]
       [0, 1, 28, 29]

    .. code-block:: c

       // WARNING: Qiskit's C API does not yet allow us to implement circuits
       // with custom gate definitions, which we therefore also cannot transpile
       // via this API.

None of the excitations sampled without filtering touch the occupied set
(``0-6`` and ``28-34``) at all, so none of them can move a particle between an
occupied and an unoccupied mode; every single one is trivial and would have
no effect on the sampled bitstrings. With ``filter_trivial=True``, all five are
rejected and replaced by excitations that do couple an occupied mode with an
unoccupied one. For example, the first accepted excitation ``[0, 1, 6, 7]`` moves a
particle between occupied modes ``0``, ``1``, and ``6`` and unoccupied mode
``7``.

Once an excitation is accepted, every mode in its support becomes
"uncertain" and, thus, eligible to play either role for later samples,
so the occupied and unoccupied mode sets keep growing as more excitations
get accepted. This is what makes the second excitation,
``[0, 1, 28, 29]``, acceptable. All four of its modes are among the
*originally* occupied ones, so it does not couple to any originally
unoccupied mode. It is only accepted because modes ``0`` and ``1`` became
uncertain (and thus eligible as the "unoccupied" side of the coupling)
once the first excitation touched them.

.. note::
   Without a preceding :class:`.InitializeModes` gate, ``filter_trivial=True``
   has no occupation information to filter against. It emits a
   :class:`UserWarning` and leaves the sampling unfiltered for that
   :class:`.Evolution` gate.

(Optional) Optimize the fermionic mode indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can add an additional optimization step to the transpilation pipeline that
minimizes the distance of the fermionic excitation spans by relabeling the
fermionic mode indices. This optimization was introduced in the `SqDRIFT`_ paper
and is implemented by :func:`.build_excitation_span_minimization_model`. It can
be easily inserted into the transpiler pipeline by using the :class:`.RelabelModes`
pass:

.. invisible-code-block: python

   >>> from qiskit_fermions.utils.optionals import HAS_PYOMO

.. skip: start if(not HAS_PYOMO)

.. tab-set-code::

    .. code-block:: python

       >>> from pyomo.environ import SolverFactory
       >>> from qiskit_fermions.transpiler.passes import RelabelModes
       >>>
       >>> solver = SolverFactory("appsi_highs")
       >>> solver.options["time_limit"] = 10
       >>>
       >>> qdrift = QDriftTrotterization(5, rng=19)
       >>> relabel = RelabelModes(solver=solver)
       >>>
       >>> pm.optimization = FermionicPassManager([qdrift, relabel])
       >>>
       >>> relabeled_circ = pm.run(circ)
       >>> # if the automatic mode relabeling was successful, the circuit's
       >>> # metadata will contain the mode `permutation` information

    .. code-block:: c

       // WARNING: This feature is not available via the C API.

.. skip: end

.. note::
   Using the automatic optimization inside :class:`.RelabelModes` (which
   leverages :func:`.build_excitation_span_minimization_model`) requires the
   optional dependency managed by :data:`.HAS_PYOMO`.

.. important::
   In order to perform the correct subspace diagonalization, the bitstrings
   sampled from circuits that were transpiled with the :class:`.RelabelModes`
   optimization pass must be post-processed based on the ``permutation``
   information contained in the circuits' metadata!

Next steps
^^^^^^^^^^

Now that you have successfully generated an ensemble of circuits, you can sample
bitstrings from them. To do so, the circuits must be executed on hardware.
Refer to the `Qiskit
documentation <https://quantum.cloud.ibm.com/docs/guides/intro-to-patterns>`_
for detailed instructions.

Once the bitstring samples have been obtained, these can be used in combination
with the Hamiltonian coefficients to perform SQD post-processing, as explained in the `SQD addon
tutorials <https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-sqd/guides/overview>`_.


.. _qDRIFT: https://arxiv.org/abs/1811.08017
.. _SQD: https://arxiv.org/abs/2405.05068
.. _SqDRIFT: https://arxiv.org/abs/2508.02578
.. _SKQD: https://arxiv.org/abs/2501.09702
