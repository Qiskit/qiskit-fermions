# Context

### *class* Context(num_qubits, , coupling_map=None, target=None, physical_qubits=None, user_edges=None)

*Where* the circuit will run.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

It holds what the high-level synthesis knows about the device, and the user’s CZ pairs.

All qubit pairs it holds and returns are in the gate’s own qubit indices. Every field but
`num_qubits` is keyword-only.

### Attributes

* **Parameters:**
  * **num_qubits** ([*int*](https://docs.python.org/3/builtins/functions.html#int))
  * **coupling_map** ([*CouplingMap*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.CouplingMap) *|* *None*)
  * **target** ([*Target*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.Target) *|* *None*)
  * **physical_qubits** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,*  *...* *]*  *|* *None*)
  * **user_edges** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *,*  *...* *]*  *|* *None*)

#### coupling_map *: [CouplingMap](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.CouplingMap) | [None](https://docs.python.org/3/builtins/constants.html#None)* *= None*

The device’s coupling map, or `None`.

#### edges

The pairs `(i, j)`, `i < j`, that a CZ gate may act on, or `None`.

It is `None` when no connectivity is known.

The user’s pairs, if given; otherwise the device’s coupling edges among the physical qubits,
if the gate is placed.

#### physical_qubits *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...] | [None](https://docs.python.org/3/builtins/constants.html#None)* *= None*

The device qubit of each qubit of the gate, or `None` before the layout stage.

#### placed

Whether the gate’s qubits sit on known device qubits of a known coupling map.

#### target *: [Target](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.Target) | [None](https://docs.python.org/3/builtins/constants.html#None)* *= None*

The device’s target, or `None`.

#### user_edges *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)], ...] | [None](https://docs.python.org/3/builtins/constants.html#None)* *= None*

CZ pairs given by the user, between qubits of the gate. They take precedence over the
device’s. The plugin maps the pairs it is given onto each gate’s qubits.

#### num_qubits *: [int](https://docs.python.org/3/builtins/functions.html#int)*

The number of qubits of the gate.
