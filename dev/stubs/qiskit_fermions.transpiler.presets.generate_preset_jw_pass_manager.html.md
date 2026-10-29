# generate_preset_jw_pass_manager

### generate_preset_jw_pass_manager(\*\*kwargs)

Generates a preset transpiler pipeline based on the [`jordan_wigner()`](qiskit_fermions.mappers.library.jordan_wigner.html.md#qiskit_fermions.mappers.library.jordan_wigner) mapping.

* **Parameters:**
  **kwargs** – any additional keyword arguments are forwarded to
  [`generate_preset_pass_manager()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.generate_preset_pass_manager) whose output is used
  for the `qubit` stage. The `hls_config` is extended, not replaced, so that a
  [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate) of a hop between two qubits is
  **always** synthesized by the [`HopEvolutionSynthesis`](qiskit_fermions.transpiler.passes.synthesis.HopEvolutionSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.HopEvolutionSynthesis), ahead of whatever methods
  the given configuration lists for it (`"default"` included, which then only applies to
  what is not a hop). The given configuration is not modified. A pipeline that should not
  do this has to be assembled from its stages rather than taken from this preset.
* **Returns:**
  The preset staged fermion-to-qubit transpiler pipeline.
* **Return type:**
  [*MultiStagePassManager*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.MultiStagePassManager)
