.. _mappers_explanation:

Mapping operator representations
================================

**Mappers** are routines that transform operators from one representation to another
while preserving mathematical equivalence. They enable you to work flexibly across
different operator algebras and prepare operators for quantum circuit execution.

Implementing custom mappers
---------------------------

The mappers module is designed to make it straightforward to implement custom
mappings. The key insight is that mappers work by transforming the fundamental
actions (creation, annihilation, Majorana, and so on) that compose an operator, then
combining these transformed actions according to the target representation's rules.

Implementing a Jordan-Wigner transformation that maps
a :class:`.MajoranaOperator` to a qubit operator using :func:`.map_majorana_action_generators` is a concrete example:

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit.quantum_info import SparsePauliOp
       >>> from qiskit_fermions.mappers import map_majorana_action_generators
       >>> from qiskit_fermions.operators import MajoranaOperator
       >>>
       >>> # Create a Majorana operator
       >>> maj_op = MajoranaOperator.from_dict({(0, 1): 1.0, (2,): 0.5})
       >>>
       >>> # Define how each Majorana action maps to Pauli strings
       >>> num_qubits = 2
       >>> def map_action(mode: int) -> SparsePauliOp:
       ...     idx = mode // 2
       ...     qubits = list(range(idx + 1))
       ...     pauli = "Y" if mode % 2 else "X"
       ...     return SparsePauliOp.from_sparse_list(
       ...         [("Z" * idx + pauli, qubits, 1.0)],
       ...         num_qubits=num_qubits,
       ...     )
       >>>
       >>> # Apply the mapping to transform the operator
       >>> sparse_pauli_op = map_majorana_action_generators(
       ...     maj_op,
       ...     map_action,
       ...     lambda: SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits=num_qubits),
       ... )
       >>> print(sparse_pauli_op.sort())
       SparsePauliOp(['II', 'IZ', 'XZ'],
                     coeffs=[0. +0.j, 0. +1.j, 0.5+0.j])

    .. code-block:: c

       // The C API provides direct mapping functions for common transformations.
       // Custom mapper prototyping as shown in Python is primarily a Python feature.
       // For custom mappings in C, you will need to perform the iteration and
       // conversion logic entirely by yourself.

The key steps to implement any custom mapper are:

1. **Define action transformation**: Specify how each action (indexed by mode) maps
   to the target representation. In the example above, Majorana actions map to Pauli
   strings according to Jordan-Wigner rules.

2. **Use one of the provided mapping functions**: the :mod:`~qiskit_fermions.mappers`
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

   Of course, you can also iterate the terms of an operator manually rather than
   using one of the provided iterator functions.

Library implementations
-----------------------

The :mod:`~qiskit_fermions.mappers.library` module provides efficient, thoroughly
tested implementations of common mappings. These follow the same underlying pattern
as custom mappers but are optimized for production use and available in both the
Python and C APIs.

Following is an example showing a library mapper applied to the same :class:`.MajoranaOperator`
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
       >>> print(SparsePauliOp.from_sparse_observable(sparse_observable).equiv(sparse_pauli_op))
       True

    .. code-block:: c

       #include <qiskit_fermions.h>

       // Create a MajoranaOperator with the same terms as maj_op above
       uint64_t num_terms = 2;
       uint64_t num_modes = 3;
       uint32_t modes[3] = {0, 1, 2};
       QkComplex67 coeffs[2] = {{1.0, 0.0}, {0.5, 0.0}};
       uint32_t boundaries[3] = {0, 2, 3};
       QfMajoranaOperator *maj_op = qf_maj_op_new(num_terms, num_modes, coeffs, modes, boundaries);

       // Convert MajoranaOperator to FermionOperator using library mapper
       QfFermionOperator *ferm_op = qf_maj_op_to_ferm_op(maj_op);

       // Apply Jordan-Wigner mapper to get qubit operator
       QfSparseObservable *qubit_op = qf_ferm_op_jordan_wigner(ferm_op, 2);

       // Clean up
       qf_ferm_op_free(ferm_op);
       qf_maj_op_free(maj_op);
       qf_sparse_obs_free(qubit_op);
