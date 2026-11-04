# HopEvolutionSynthesis

### *class* HopEvolutionSynthesis(, atol=1e-12)

A high-level synthesis plugin for the evolution under a hop between two qubits.

Bases: [`HighLevelSynthesisPlugin`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin)

A [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate) whose operator is a hop (see
`match_hop()`), such as the image of a fermionic hop between adjacent modes under
Jordan-Wigner, is synthesized as a single [`XXPlusYYGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.XXPlusYYGate). That is
the exact evolution. The generic synthesis of the same operator costs several times as many
two-qubit gates, and a target whose only two-qubit gate is the
[`XXPlusYYGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.XXPlusYYGate) cannot be reached by it at all.

Any other gate is declined (the plugin returns `None`), so that the next method of the
[`HLSConfig`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.HLSConfig) is tried.

#### NOTE
The `synthesis` of a matched gate does not
apply: the result is exact, whereas a product formula approximates the evolution of a hop
with a complex amplitude (its $XX + YY$ and $XY - YX$ parts do not commute). A
matched gate therefore differs from the generic synthesis by that Trotter error, and is the
more accurate of the two.

Register it ahead of the default method:

```default
HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])
```

The [`generate_preset_jw_pass_manager()`](qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager.html.md#qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager) does so unconditionally, and merges the entry into
any `hls_config` passed to it. That mirrors how the preset has always lowered an orbital
rotation: it emits an [`XXPlusYYGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.XXPlusYYGate) for each Givens factor, and
the configuration cannot turn that off.

Initializing this plugin can be done with the arguments listed below.

* **Parameters:**
  **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – the absolute tolerance on the coefficients of the operator, within which it is
  still considered a hop.

### Attributes

#### atol

The absolute coefficient tolerance with which an operator is recognized as a hop.

### Methods

#### run(high_level_object, coupling_map=None, target=None, qubits=None, \*\*options)

Runs this plugin.

* **Parameters:**
  * **high_level_object** – the operation to synthesize.
  * **coupling_map** – unused.
  * **target** – unused.
  * **qubits** – unused.
  * **options** – unused.
* **Returns:**
  A circuit with a single [`XXPlusYYGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.XXPlusYYGate) if
  `high_level_object` is a [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate) of a hop
  with a fixed evolution time, and `None` otherwise.
