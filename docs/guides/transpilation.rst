.. _transpilation_explanation:

Transpiling fermionic circuits
==============================

.. important::

   The concepts in this guide are currently available only in the Python API.
   Equivalent functionality will be made available via the C API in a future release.

This guide explains how to transpile a :class:`.FermionicCircuit` to a standard
:class:`~qiskit.circuit.QuantumCircuit`. We continue with the same time evolution
example from the :ref:`fermionic circuits guide <fermionic_circuit_explanation>`.

Transpilation stages
--------------------

As explained in :mod:`qiskit_fermions.transpiler`, the preset pass managers in
this project split the transpilation into the following stages:

**Input**
   Converts the input circuit to a DAG data structure.

**Optimization**
   Fermionic-level optimizations that keep the circuit in fermionic space. These
   passes exploit fermionic structure and commutation relations while
   problem-aware knowledge remains fully available.

**Layout**
   Maps fermionic mode registers to quantum registers. For occupation-basis
   mappings like Jordan-Wigner, each fermionic mode maps to a single qubit. This
   stage also supports more general mappings where the number of qubits differs
   from the number of fermionic modes.

**Synthesis**
   Converts fermionic gates to qubit operations by using the chosen fermion-to-qubit
   mapping. :class:`.FermionicGate` instances are transformed into sequences of
   standard quantum gates.

**Qubit**
   Standard Qiskit transpilation on the resulting qubit circuit, including
   optimization, layout, and routing for your target hardware.

**Output**
   Converts the internal DAG data structure to the desired output format.

A practical example: transpilation using Jordan-Wigner
------------------------------------------------------

Use the preset pass manager to transpile your fermionic circuit with the
Jordan-Wigner fermion-to-qubit mapping:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. plot::
         :context:
         :nofigs:
         :include-source:

         >>> from qiskit_fermions.circuit import FermionicCircuit
         >>> from qiskit_fermions.circuit.library import Evolution
         >>> from qiskit_fermions.operators import FermionOperator, cre, ann
         >>> from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
         >>>
         >>> # Create the same fermionic circuit from the previous guide
         >>> circuit = FermionicCircuit(4)
         >>> hamiltonian = FermionOperator.from_terms([
         ...     ([cre(0), ann(2)], 0.5),
         ...     ([cre(2), ann(0)], 0.5),
         ...     ([cre(1), ann(3)], 0.5),
         ...     ([cre(3), ann(1)], 0.5),
         ... ])
         >>> hamiltonian.groups = [0, 0, 1, 1]
         >>> evolution = Evolution(4, hamiltonian, time=1.0)
         >>> circuit.append(evolution, circuit.register)
         >>>
         >>> # Generate the Jordan-Wigner transpilation pipeline
         >>> pm = generate_preset_jw_pass_manager()
         >>>
         >>> # Transpile the fermionic circuit to a qubit circuit
         >>> qubit_circuit = pm.run(circuit)
         >>>
         >>> # The result is a standard QuantumCircuit
         >>> print(type(qubit_circuit))
         <class 'qiskit.circuit.quantumcircuit.QuantumCircuit'>

   .. tab-item:: C
      :sync: c

      .. code-block:: c

         // The C API for transpilation will be made available in a future release.

.. plot::
   :alt: The transpiled QuantumCircuit of the fermionic time evolution.
   :context: close-figs

   >>> # Draw the transpiled circuit
   >>> qubit_circuit.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

Compare with the traditional workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :ref:`fermionic circuits <fermionic_circuit_explanation>` guide describes
how the traditional workflow performs fermion-to-qubit encoding *before*
building the :class:`~qiskit.circuit.QuantumCircuit`. Compare the two
approaches:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.mappers.library import jordan_wigner
   >>> from qiskit.circuit import QuantumCircuit
   >>> from qiskit.circuit.library import PauliEvolutionGate
   >>>
   >>> # Map the fermionic Hamiltonian to a qubit operator
   >>> qubit_hamiltonian = jordan_wigner(hamiltonian, 4).simplify()
   >>>
   >>> # Build the QuantumCircuit directly
   >>> quantum_circuit = QuantumCircuit(4)
   >>> pauli_evolution = PauliEvolutionGate(qubit_hamiltonian, time=1.0)
   >>> quantum_circuit.append(pauli_evolution, quantum_circuit.qubits)
   <qiskit.circuit.instructionset.InstructionSet object at ...>

.. plot::
   :alt: The directly constructed QuantumCircuit of the fermionic time evolution.
   :context: close-figs

   >>> quantum_circuit.decompose().draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

Both workflows produce equivalent circuits. However, the traditional workflow
requires you to implement optimization steps manually, whereas the
fermionic-first approach allows you to integrate problem-aware optimizations
into the transpilation process.

Advanced workflows
------------------

The preset pass manager provides a convenient starting point. For more advanced
use cases, you can:

- Customize the transpiler passes run during each stage.
- Implement custom fermion-to-qubit mappings by creating synthesis plugins (see
  :class:`.F2QSynthesisPlugin`).
- Build custom transpiler passes.
