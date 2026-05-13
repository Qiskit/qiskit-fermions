.. _sqdrift_tutorial:

Generate SqDRIFT circuits
=========================

With `SQD`_, you must choose an ansatz from which to sample bitstrings.  
The `SqDRIFT`_ variant uses an ensemble of time-evolution circuits
constructed directly from the target Hamiltonian instead.
This is achieved by subsampling smaller time-evolution operators from the
Hamiltonian based on its coefficients, which is known as the `qDRIFT`_
Trotterization method.

This tutorial shows how to generate an ensemble of randomized circuits as described.

1. Hamiltonian setup
^^^^^^^^^^^^^^^^^^^^

For the purposes of this tutorial, we load the electronic structure Hamiltonian
of N2 from an FCIDUMP file. Of course, there are other means of
constructing the :class:`.FermionOperator`. Be sure to check out its
documentation, as well as the :mod:`qiskit_fermions.operators.library`.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.operators.library import FCIDump
       >>> from qiskit_fermions.operators import FermionOperator
       >>>
       >>> fcidump = FCIDump.from_file("docs/tutorials/n2.fcidump")
       >>> num_modes = 2 * fcidump.norb
       >>> hamil = FermionOperator.from_fcidump(fcidump)

    .. code-block:: c

       #include <qiskit_fermions.h>

       QfFCIDump* fcidump = qf_fcidump_from_file("docs/tutorials/n2.fcidump");
       QfFermionOperator* hamil = qf_ferm_op_from_fcidump(fcidump);
       uint32_t num_modes = 2 * qf_fcidump_norb(fcidump);


2. Group Hamiltonian terms
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this step, we exploit the many symmetries that are present in the electronic
structure Hamiltonian by grouping related terms that have identical coefficients.
This action changes the operator coefficient distribution that the qDRIFT
protocol samples from, but it does not affect its convergence guarantees.
Crucially, grouping terms that are related by symmetry results in a favorable
cancellation of Pauli terms, resulting in an overall shorter circuit depth when
time-evolving a state under their action.

The :mod:`qiskit_fermions.operators.grouping` module provides convenience
functions for grouping an operator's terms. This is explained in more
detail in :ref:`this guide <grouping_explanation>`.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.operators.grouping import group_terms_by_electronic_structure
       >>>
       >>> exit_code = group_terms_by_electronic_structure(hamil, num_modes)
       >>> assert exit_code is None

    .. code-block:: c

       QfExitCode exit = qf_group_terms_by_electronic_structure(hamil, num_modes, false);

3. Prepare the time evolution circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this step, we prepare the Hamiltonian's time evolution circuit and the 
base circuit from which to draw samples. The
:mod:`qiskit_fermions.circuit.library` contains all the required
components to do so, in compliance with Qiskit conventions.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.circuit import FermionicCircuit
       >>> from qiskit_fermions.circuit.library import Evolution
       >>>
       >>> time = 1.0  # you can choose a desired scaling factor here
       >>> evo_gate = Evolution(num_modes, hamil, time)
       >>>
       >>> circ = FermionicCircuit(num_modes)
       >>> circ.append(evo_gate, circ.modes)

    .. code-block:: c

       // WARNING: Qiskit's C API does not yet allow us to implement circuits
       // with custom gate definitions.

.. note::
   In this example, we neither initialize the fermionic modes with particles,
   nor measure their final state.

4. Transpile the circuit with QDrift Trotterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`qiskit_fermions.transpiler` module integrates directly with Qiskit's
transpilation pipeline, allowing the :class:`.FermionicCircuit` constructed above
to be directly transpiled to a :external:class:`~qiskit.circuit.QuantumCircuit`.

Here, we are using the :func:`.jordan_wigner` fermion-to-qubit mapping to
convert the Hamiltonian expressed in terms of fermions to be expressed in Pauli strings instead. This
can be done directly as part of the transpilation process by using the
:class:`.EvolutionSynthesis` transpilation pass plugin. Here, we are using
:func:`.generate_preset_jw_pass_manager` to build 
:class:`.FermionicStagedPassManager`, which ensures that 
Jordan-Wigner encoding is uses consistently for all circuit instructions.

Crucially, we add the :class:`.QDriftTrotterization` transpilation pass to the
``optimization`` stage of the transpilation pipeline. This ensures that we do
not use the time evolution of the entire Hamiltonian, a circuit whose depth
would exceed the capabilities of currently available quantum computing hardware.

Instead, it will subsample a fixed number of ``groups`` of Hamiltonian terms for
each circuit, every time we transpile the circuit. Through this, we can generate
multiple circuit randomizations as required by the `SqDRIFT`_ algorithm by
repeatedly running the transpilation pipeline.

This step also introduces the few parameters with which one can tweak the
ensemble of circuits to generate:

* the number of circuits to generate: ``num_sqdrift_randomizations``
* the length of each circuit in terms of excitation groups: ``num_groups``

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.transpiler import FermionicPassManager
       >>> from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
       >>> from qiskit_fermions.transpiler.passes import QDriftTrotterization
       >>>
       >>> num_groups = 10
       >>> qdrift = QDriftTrotterization(num_groups, rng=42)
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
   In the example above we have fixed the ``seed`` for the random number
   generator used inside of the :class:`.QDriftTrotterization` transpilation
   pass.

Next steps
^^^^^^^^^^

Now that we have successfully generated an ensemble of circuits, we can sample
bitstrings from them. To do so, the circuits must be executed on hardware. 
Refer to the `Qiskit
documentation <https://quantum.cloud.ibm.com/docs/guides/intro-to-patterns>`_
for detailed instructions.

Once the bitstring samples have been obtained, these can be used in combination
with the Hamiltonian coefficients to perform SQD post-processing, as explained in the `SQD addon tutorials
<https://qiskit.github.io/qiskit-addon-sqd/tutorials/index.html>`_ tutorial.


.. _qDRIFT: https://arxiv.org/abs/1811.08017
.. _SQD: https://arxiv.org/abs/2405.05068
.. _SqDRIFT: https://arxiv.org/abs/2508.02578
.. _SKQD: https://arxiv.org/abs/2501.09702
