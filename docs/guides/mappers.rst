.. _mappers_explanation:

Map operator representations
============================

**Mappers** are routines that transform operators from one representation to another
while preserving mathematical equivalence. Use them to work flexibly across
different operator algebras and to prepare operators for quantum circuit execution.

Implement custom mappers
------------------------

The mappers module makes it straightforward to implement custom
mappings. Mappers work by transforming the fundamental
actions (creation, annihilation, Majorana, and so on) that compose an operator, then
combining these transformed actions according to the target representation's rules.

As a concrete example, consider a Jordan-Wigner transformation that maps
a :class:`.MajoranaOperator` to a qubit operator using :func:`.map_majorana_action_generators`:

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit.quantum_info import SparseObservable
       >>> from qiskit_fermions.mappers import map_majorana_action_generators
       >>> from qiskit_fermions.operators import MajoranaOperator
       >>>
       >>> # Create a Majorana operator
       >>> maj_op = MajoranaOperator.from_dict({(0, 1): 1.0, (2,): 0.5})
       >>>
       >>> # Define how each Majorana action maps to Pauli strings
       >>> num_qubits = 2
       >>> def map_action(mode: int) -> SparseObservable:
       ...     idx = mode // 2
       ...     qubits = list(range(idx + 1))
       ...     pauli = "Y" if mode % 2 else "X"
       ...     return SparseObservable.from_sparse_list(
       ...         [("Z" * idx + pauli, qubits, 1.0)],
       ...         num_qubits=num_qubits,
       ...     )
       >>>
       >>> # Apply the mapping to transform the operator
       >>> custom_observable = map_majorana_action_generators(
       ...     maj_op,
       ...     map_action,
       ...     identity=lambda: SparseObservable.identity(num_qubits),
       ...     compose=SparseObservable.compose,
       ... )
       >>> print(custom_observable.simplify())
       <SparseObservable with 2 terms on 2 qubits: (0+1j)(Z_0) + (0.5+0j)(X_1 Z_0)>

    .. code-block:: c

       // The C API provides direct mapping functions for common transformations.
       // Custom mapper prototyping as shown in Python is primarily a Python feature.
       // For custom mappings in C, you will need to perform the iteration and
       // conversion logic entirely by yourself.

The key steps to implement any custom mapper are:

1. **Define action transformation**: Specify how each action (indexed by mode) maps
   to the target representation. In the example above, Majorana actions map to Pauli
   strings according to Jordan-Wigner rules.

2. **Use one of the provided mapping functions**: The :mod:`~qiskit_fermions.mappers`
   module provides utility functions to handle iterating over the operator terms and
   subsequent applications of the custom action map for the operator
   representations provided by the :mod:`~qiskit_fermions.operators` module. For example,
   :func:`.map_majorana_action_generators` works with Majorana operators.

3. **Get the result**: The mapping function returns an operator in the target
   representation with the transformation applied consistently across all terms.

This design makes it easy to prototype custom mappings without deep knowledge of
internal operator structures. The transformation is applied automatically to every
term in the operator while preserving mathematical equivalence.

.. hint::

   You can also iterate the terms of an operator manually rather than
   using one of the provided iterator functions. See the section on `term iteration
   <term_iteration_and_reconstruction>`_ in the operators guide for details.

Library implementations
-----------------------

The :mod:`qiskit_fermions.mappers.library` module provides efficient, thoroughly
tested implementations of common mappings. These follow the same underlying pattern
as custom mappers but are optimized for production use and are available in both the
Python and C APIs.

The example below applies a library mapper to the same :class:`.MajoranaOperator`
from the custom mapper section, demonstrating equivalent results:

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.mappers.library import majorana_to_fermion, jordan_wigner
       >>>
       >>> # Convert MajoranaOperator to FermionOperator using library mapper
       >>> ferm_op = majorana_to_fermion(maj_op)
       >>>
       >>> # Apply Jordan-Wigner mapper to get qubit operator
       >>> sparse_observable = jordan_wigner(ferm_op, num_qubits=2)
       >>> difference = sparse_observable - custom_observable
       >>> print(difference.simplify() == SparseObservable.zero(num_qubits))
       True

    .. code-block:: c

       #include <qiskit_fermions.h>

       // Create a MajoranaOperator with the same terms as maj_op above
       uint64_t num_terms = 2;
       uint64_t num_modes = 3;
       uint32_t modes[3] = {0, 1, 2};
       QkComplex64 coeffs[2] = {{1.0, 0.0}, {0.5, 0.0}};
       uint32_t boundaries[3] = {0, 2, 3};
       QfMajoranaOperator *maj_op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

       // Convert MajoranaOperator to FermionOperator using library mapper
       QfFermionOperator *ferm_op = qf_maj_op_to_ferm_op(maj_op);

       // Apply Jordan-Wigner mapper to get qubit operator
       QkObs *qubit_op;
       QfExitCode exit = qf_ferm_op_jordan_wigner(ferm_op, 2, &qubit_op);
       assert(exit == QfExitCode_Success);

       // Clean up
       qf_ferm_op_free(ferm_op);
       qf_maj_op_free(maj_op);
       qk_obs_free(qubit_op);

.. note::
   The two-step above is written out to show two library mappers composing. In practice you would map
   a :class:`.MajoranaOperator` in one step instead, with :func:`.majorana_jordan_wigner`
   (:c:func:`qf_maj_op_jordan_wigner` in the C API), and likewise for the other operator types --
   :func:`.edge_vertex_jordan_wigner` and :func:`.transfer_vertex_jordan_wigner`
   (:c:func:`qf_edge_op_jordan_wigner`, :c:func:`qf_transfer_op_jordan_wigner`). Going via a
   :class:`.FermionOperator` also inflates the intermediate result: every generator of these algebras
   maps onto a single Pauli string, whereas each fermionic action maps onto a two-term sum, so a term
   built from several generators expands only to be merged back down again. That saving grows with the
   length of the terms; for the single-generator terms of a hopping Hamiltonian the two routes cost
   about the same.

In the Python API, some of these library mappers are also reachable through type-agnostic
convenience mappers. :func:`.fermion_operator` and :func:`.majorana_operator` sit in front of
:func:`.majorana_to_fermion` and :func:`.fermion_to_majorana` (among others), delegating to the
appropriate library implementation internally based on the type of object they are given,
rather than requiring you to pick the right function yourself. The same applies to
:func:`.jordan_wigner`, which dispatches to whichever of the four direct implementations matches the
operator it is given. See :mod:`qiskit_fermions.protocols` for how this dispatch is implemented
internally.
