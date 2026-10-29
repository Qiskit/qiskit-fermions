# generate_preset_jw_pass_manager

### generate_preset_jw_pass_manager(\*\*kwargs)

Generates a preset transpiler pipeline based on the [`jordan_wigner()`](qiskit_fermions.mappers.library.jordan_wigner.html.md#qiskit_fermions.mappers.library.jordan_wigner) mapping.

* **Parameters:**
  **kwargs** – any additional keyword arguments are forwarded to
  [`generate_preset_pass_manager()`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.generate_preset_pass_manager) whose output is used
  for the `qubit` stage.
* **Returns:**
  The preset staged fermion-to-qubit transpiler pipeline.
* **Return type:**
  [*MultiStagePassManager*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.MultiStagePassManager)
