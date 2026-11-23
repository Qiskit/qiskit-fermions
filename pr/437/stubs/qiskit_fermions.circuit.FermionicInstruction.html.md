# FermionicInstruction

### *class* FermionicInstruction(name, num_qubits, num_clbits, params, label=None)

The base class for all instructions acting on fermionic modes.

Bases: [`Instruction`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Instruction)

This is the type that the fermion-to-qubit synthesis stage dispatches on, so it is what makes an
instruction “fermionic” as far as this package is concerned.

Most fermionic instructions are unitary and should subclass [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) instead, which
adds the [`Gate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Gate) interface on top of this one. Subclass this class directly
only for an instruction that cannot be a gate, the motivating case being one that acts on
classical bits: [`Gate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Gate) fixes its classical-bit count at zero, so a
measurement ([`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure)) cannot be a gate.

#### CAUTION
Since this is a subclass of [`Instruction`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Instruction) the documentation of its
methods may refer to qubits. Those references should be interpreted as referring to
fermions in the context of instances of this subclass.

It may also happen that some of the inherited methods may not always make sense because of
this re-interpretation. You have been warned.

* **Parameters:**
  * **name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – instruction name
  * **num_qubits** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – instruction’s qubit width
  * **num_clbits** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – instruction’s clbit width
  * **params** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *|*[*float*](https://docs.python.org/3/builtins/functions.html#float) *|*[*complex*](https://docs.python.org/3/builtins/functions.html#complex) *|*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *|**ndarray* *|*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *|**ParameterExpression* *]*) – list of parameters
  * **label** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *or* *None*) – An optional label for identifying the instruction.
* **Raises:**
  * **CircuitError** – when the register is not in the correct format.
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – when the optional label is provided, but it is not a string.

### Attributes

#### num_modes

The number of fermionic modes that this instruction acts upon.

### Inherited Attributes

#### base_class

Get the base class of this instruction.  This is guaranteed to be in the inheritance tree
of `self`.

The “base class” of an instruction is the lowest class in its inheritance tree that the
object should be considered entirely compatible with for \_all_ circuit applications.  This
typically means that the subclass is defined purely to offer some sort of programmer
convenience over the base class, and the base class is the “true” class for a behavioral
perspective.  In particular, you should *not* override [`base_class`](#qiskit_fermions.circuit.FermionicInstruction.base_class) if you are
defining a custom version of an instruction that will be implemented differently by
hardware, such as an alternative measurement strategy, or a version of a parametrized gate
with a particular set of parameters for the purposes of distinguishing it in a
`Target` from the full parametrized gate.

This is often exactly equivalent to `type(obj)`, except in the case of singleton instances
of standard-library instructions.  These singleton instances are special subclasses of their
base class, and this property will return that base.  For example:

```default
>>> isinstance(XGate(), XGate)
True
>>> type(XGate()) is XGate
False
>>> XGate().base_class is XGate
True
```

In general, you should not rely on the precise class of an instruction; within a given
circuit, it is expected that `Instruction.name` should be a more suitable
discriminator in most situations.

#### decompositions

Get the decompositions of the instruction from the SessionEquivalenceLibrary.

#### definition

Return definition in terms of other basic gates.

#### label

Return instruction label

#### mutable

Is this instance is a mutable unique instance or not.

If this attribute is `False` the gate instance is a shared singleton
and is not mutable.

#### name

Return the name.

#### num_clbits

Return the number of clbits.

#### num_qubits

Return the number of qubits.

#### params

The parameters of this `Instruction`.  Ideally these will be gate angles.

### Inherited Methods

#### add_decomposition(decomposition)

Add a decomposition of the instruction to the SessionEquivalenceLibrary.

#### broadcast_arguments(qargs, cargs)

Validation of the arguments.

* **Parameters:**
  * **qargs** (*List*) – List of quantum bit arguments.
  * **cargs** (*List*) – List of classical bit arguments.
* **Yields:**
  *Tuple(List, List)* – A tuple with single arguments.
* **Raises:**
  **CircuitError** – If the input is not valid. For example, the number of
      arguments does not match the gate expectation.

#### copy(name=None)

Copy of the instruction.

* **Parameters:**
  **name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – name to be given to the copied circuit, if `None` then the name stays the same.
* **Returns:**
  a copy of the current instruction, with the name updated if it
  was provided
* **Return type:**
  [qiskit.circuit.Instruction](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Instruction)

#### inverse(annotated=False)

Invert this instruction.

If annotated is False, the inverse instruction is implemented as
a fresh instruction with the recursively inverted definition.

If annotated is True, the inverse instruction is implemented as
`AnnotatedOperation`, and corresponds to the given instruction
annotated with the “inverse modifier”.

Special instructions inheriting from Instruction can
implement their own inverse (e.g. T and Tdg, Barrier, etc.)
In particular, they can choose how to handle the argument `annotated`
which may include ignoring it and always returning a concrete gate class
if the inverse is defined as a standard gate.

* **Parameters:**
  **annotated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – if set to True the output inverse gate will be returned
  as `AnnotatedOperation`.
* **Returns:**
  The inverse operation.
* **Raises:**
  **CircuitError** – if the instruction is not composite
      and an inverse has not been implemented for it.

#### is_parameterized()

Return whether the `Instruction` contains [compile-time parameters](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit#circuit-compile-time-parameters).

#### repeat(n)

Creates an instruction with `self` repeated $n$ times.

* **Parameters:**
  **n** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Number of times to repeat the instruction
* **Returns:**
  Containing the definition.
* **Return type:**
  [qiskit.circuit.Instruction](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Instruction)
* **Raises:**
  **CircuitError** – If n < 1.

#### reverse_ops()

For a composite instruction, reverse the order of sub-instructions.

This is done by recursively reversing all sub-instructions.
It does not invert any gate.

* **Returns:**
  a new instruction with
  : sub-instructions reversed.
* **Return type:**
  [qiskit.circuit.Instruction](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.Instruction)

#### soft_compare(other)

Soft comparison between gates. Their names, number of qubits, and classical
bit numbers must match. The number of parameters must match. Each parameter
is compared. If one is a ParameterExpression then it is not taken into
account.

* **Parameters:**
  **other** (*instruction*) – other instruction.
* **Returns:**
  are self and other equal up to parameter expressions.
* **Return type:**
  [bool](https://docs.python.org/3/builtins/functions.html#bool)

#### to_mutable()

Return a mutable copy of this gate.

This method will return a new mutable copy of this gate instance.
If a singleton instance is being used this will be a new unique
instance that can be mutated. If the instance is already mutable it
will be a deepcopy of that instance.

#### validate_parameter(parameter)

Instruction parameter has no validation or normalization.
