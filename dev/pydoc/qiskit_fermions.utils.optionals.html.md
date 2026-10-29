<a id="module-qiskit_fermions.utils.optionals"></a>

# Optional Dependencies

This module defines lazy availability testers for optional third-party dependencies. The checkers
are instances of [`LazyImportTester`](https://quantum.cloud.ibm.com/docs/api/qiskit/utils#qiskit.utils.LazyImportTester) and can be used as booleans
(evaluated lazily), or to raise a [`MissingOptionalLibraryError`](https://quantum.cloud.ibm.com/docs/api/qiskit/exceptions#qiskit.exceptions.MissingOptionalLibraryError)
when a dependency is required.

## Available Testers

### HAS_FFSIM *= <qiskit.utils.lazy_tester.LazyImportTester object>*

[ffsim](https://github.com/qiskit-community/ffsim) is a high-performance simulator for
fermionic quantum circuits that exploits particle-number and spin-Z conservation.

[`FermionicCircuit`](../stubs/qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) instances can be simulated using
[`ffsim.apply_unitary()`](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.apply_unitary.html#ffsim.apply_unitary).

#### NOTE
`ffsim` does not support Windows, which is why it is an optional dependency.

#### SEE ALSO
[`LazyDependencyManager`](https://quantum.cloud.ibm.com/docs/api/qiskit/utils#qiskit.utils.LazyDependencyManager) for usage examples and the available
methods of this object.

### HAS_PYOMO *= <qiskit.utils.lazy_tester.LazyImportTester object>*

[Pyomo](https://www.pyomo.org/) is a Python-based optimization modeling language used to build
linear and mixed-integer programs (LP/MILP) for classical solvers.

#### SEE ALSO
[`LazyDependencyManager`](https://quantum.cloud.ibm.com/docs/api/qiskit/utils#qiskit.utils.LazyDependencyManager) for usage examples and the available
methods of this object.

### HAS_QISKIT_ADDON_SQD *= <qiskit.utils.lazy_tester.LazyImportTester object>*

[qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) implements sample-based quantum
diagonalization (SQD): it projects a Hamiltonian onto the subspace spanned by measured computational-
basis configurations and iteratively refines that subspace via *configuration recovery*.

#### NOTE
`qiskit-addon-sqd` pulls in PySCF, which does not support Windows, which is why it is an optional
dependency.

#### SEE ALSO
[`LazyDependencyManager`](https://quantum.cloud.ibm.com/docs/api/qiskit/utils#qiskit.utils.LazyDependencyManager) for usage examples and the available
methods of this object.
