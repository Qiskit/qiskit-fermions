# FermionicPassManager

### *class* FermionicPassManager(tasks=(), max_iteration=1000)

A transpiler pass manager converting one [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit) to another.

Bases: [`BasePassManager`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.BasePassManager)[[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)]

The internal representation (IR) of this pass manager is [`FermionicDAGCircuit`](qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit), but the
input and output type are [`FermionicCircuit`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit).

Initialize an empty pass manager object.

* **Parameters:**
  * **tasks** (*Task* *[**IR* *,* *IR* *]*  *|* [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[**Task* *[**IR* *,* *IR* *]* *]*) – A pass set to be added to the pass manager schedule.
  * **max_iteration** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – The maximum number of iterations the schedule will be looped if the
    condition is not met.

### Inherited Methods

#### append(tasks)

Append tasks to the schedule of passes.

* **Parameters:**
  **tasks** ([*Task*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.Task) *[**IR* *,* *IR* *]*  *|* [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*Task*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.Task) *[**IR* *,* *IR* *]* *]*) – A set of pass manager tasks to be added to schedule.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – When any element of tasks is not a subclass of passmanager Task.
* **Return type:**
  None

#### remove(index)

Removes a particular pass in the scheduler.

* **Parameters:**
  **index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Pass index to remove, based on the position in [`passes()`](../pydoc/qiskit_fermions.transpiler.passes.html.md#module-qiskit_fermions.transpiler.passes).
* **Raises:**
  **PassManagerError** – If the index is not found.
* **Return type:**
  None

#### replace(index, tasks)

Replace a particular pass in the scheduler.

* **Parameters:**
  * **index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Task index to replace, based on the position in `tasks()`
  * **tasks** ([*Task*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.Task) *[**IR* *,* *IR* *]*  *|* [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*Task*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.Task) *[**IR* *,* *IR* *]* *]*) – A set of pass manager tasks to be added to schedule.
* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – When any element of tasks is not a subclass of passmanager Task.
  * **PassManagerError** – If the index is not found.
* **Return type:**
  None

#### run(in_programs, callback=None, num_processes=None, , property_set=None, \*\*kwargs)

Run all the passes on the specified `in_programs`.

* **Parameters:**
  * **in_programs** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any) *|* [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any) *]*) – Input programs to transform via all the registered passes.
    A single input object cannot be a Python builtin list object.
    A list object is considered as multiple input objects to optimize.
  * **callback** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[*[*Task*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.Task) *,* *IR* *,* [*PropertySet*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PropertySet) *,* [*float*](https://docs.python.org/3/builtins/functions.html#float) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *,* *None* *]*  *|* *None*) – 

    A callback function that will be called after each pass execution. The
    function will be called with 5 keyword arguments:
    ```default
    task (GenericPass): the pass being run
    passmanager_ir (Any): depending on pass manager subclass
    property_set (PropertySet): the property set
    running_time (float): the time to execute the pass
    count (int): the index for the pass execution
    ```

    The exact arguments passed expose the internals of the pass
    manager and are subject to change as the pass manager internals
    change. If you intend to reuse a callback function over
    multiple releases be sure to check that the arguments being
    passed are the same.

    To use the callback feature you define a function that will
    take in kwargs dict and access the variables. For example:
    ```default
    def callback_func(**kwargs):
        task = kwargs['task']
        passmanager_ir = kwargs['passmanager_ir']
        property_set = kwargs['property_set']
        running_time = kwargs['running_time']
        count = kwargs['count']
        ...
    ```
  * **num_processes** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *|* *None*) – The maximum number of parallel processes to launch if parallel
    execution is enabled. This argument overrides `num_processes` in the user
    configuration file, and the `QISKIT_NUM_PROCS` environment variable. If set
    to `None` the system default or local user configuration will be used.
  * **property_set** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *,* [*object*](https://docs.python.org/3/builtins/functions.html#object) *]*  *|* *None*) – If given, the initial value to use as the `PropertySet` for the
    pass manager pipeline.  This can be used to persist analysis from one run to
    another, in cases where you know the analysis is safe to share.  Beware that some
    analysis will be specific to the input circuit and the particular `Target`,
    so you should take a lot of care when using this argument.
  * **kwargs** – Arbitrary arguments passed to the compiler frontend and backend.
* **Returns:**
  The transformed program(s).
* **Return type:**
  [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)

#### to_flow_controller()

Linearize this manager into a single `FlowControllerLinear`,
so that it can be nested inside another pass manager.

* **Returns:**
  A linearized pass manager.
* **Return type:**
  [*FlowControllerLinear*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.FlowControllerLinear)[*IR*, *IR*]
