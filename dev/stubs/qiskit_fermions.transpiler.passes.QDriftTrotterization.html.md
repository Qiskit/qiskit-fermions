# QDriftTrotterization

### *class* QDriftTrotterization(\*args, \*\*kwargs)

A transpilation pass to Trotterize [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates via qDRIFT, skipping trivial draws.

Bases: [`GenericPass`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.GenericPass)[[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit), [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)]

The qDRIFT protocol itself is the synthesis method [`FermionicQDrift`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift), which is on an
equal footing with the other product formulas and is used through the [`F2FSynthesis`](qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis)
pass. This pass adds the one thing a synthesis method cannot do, since it sees a single gate
at a time: it tracks which modes are known to be occupied or unoccupied through the circuit, and
rejects a drawn term unless its support couples a known-occupied mode with a known-unoccupied
one, drawing a replacement in its place.

The method that draws the terms of an [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate is resolved per gate. A gate whose
[`synthesis`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.synthesis) is a [`FermionicQDrift`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift) (or a subclass of it) is drawn
with that method, which is how a gate carries its own `num_terms`, `weights` and `rng`.
Any other [`synthesis`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.synthesis) is not a statement about qDRIFT and is ignored. Such
a gate is drawn with the `qdrift` method given to this pass instead, and when there is none, it
is left in place (with a [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning)).

That exists to stop the sampling from spending one of the `num_terms` slots on an excitation
that cannot move a particle, and which therefore tells you nothing about the sampled bitstrings.
It buys that with the protocol’s convergence guarantee. To sample without rejecting anything,
use [`FermionicQDrift`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift) with [`F2FSynthesis`](qiskit_fermions.transpiler.passes.F2FSynthesis.html.md#qiskit_fermions.transpiler.passes.F2FSynthesis) instead.

#### WARNING
Filtering does not make the circuit shorter. `num_terms` is fixed, so a rejected draw is
replaced rather than dropped, and the cheap term it would have contributed (a diagonal
rotation, say) gives way to a coupling excitation that costs more to synthesize. Expect the
filtered circuit to be deeper than the unfiltered one: the budget of sampled slots is what the
filtering conserves, not the depth.

Rejecting a drawn term and re-drawing renormalizes the sampling distribution over the
accepted terms only, so the sampled product no longer averages to $e^{-i t H}$ for the
Hamiltonian you passed in: the effective coefficient of every retained term is inflated by
the reciprocal of the acceptance probability, and the rejected terms drop out. Neither
$\lambda$ nor $\delta$ is adjusted to compensate, so the distortion does not
cancel.

Use it only when the sampled bitstrings are the quantity of interest, as in the SqDRIFT
workflow of [Generate SqDRIFT circuits](../guides/sqdrift.html.md#sqdrift-getting-started). Do not use it when the sampled circuits are used
to estimate an expectation value, a time-evolved observable, or anything else that relies
on the qDRIFT error bound: those results are biased by an amount the pass does not track.

Filtering requires an [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) or [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) gate to
precede the [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates being Trotterized, to seed the initial occupied and
unoccupied mode sets. If none is found, or if the sets it seeds turn out to be entirely occupied
or entirely unoccupied, filtering is skipped for that gate and a [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning) is emitted
instead, and the gate is sampled as [`FermionicQDrift`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift) would.

The sets are updated as the circuit is traversed: a drawn term is accepted only if its support
intersects *both* sets, and once it is, every mode in its support becomes “uncertain” and is
added to *both* sets, making it eligible to participate in either role for subsequent draws. Any
[`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) gate does the same to every mode it acts on, since the rotation can
mix it with any other mode in its support. A [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) gate is treated
as its [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) and [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) components applied back-to-back:
its `occupation` first seeds the sets, and then every mode it acts on is marked “uncertain”.

### Filtering diagnostics

When the filtering actually filters a gate, the returned [`FermionicDAGCircuit`](qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit) records
how many draws it discarded in its [`metadata`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit#metadata), under
`filter_trivial.discarded` and `filter_trivial.emitted`. Both hold one entry per filtered
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate, in circuit order. Their ratio estimates the acceptance probability,
and hence the factor by which the filtering inflated the coefficients of the terms it kept: an
acceptance probability close to one means the filtering barely moved the distribution, while a
small one means the retained terms were weighted far above their true share of the Hamiltonian.

#### IMPORTANT
Neither field is present when no gate was filtered, which covers every case in which
filtering was skipped with a [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning). Read them defensively, for example with
`qcirc.metadata.get("filter_trivial.discarded")`. A discarded count of zero is different
from an absent field: it says the filtering ran on that gate and accepted every draw, so it
left the sampling distribution untouched.

Initializing this transpiler pass can be done with the argument below.

* **Parameters:**
  * **qdrift** – the synthesis method that draws the terms of every [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate
    that does not carry a [`FermionicQDrift`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift) as its own
    [`synthesis`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate.synthesis). Its `num_terms`, `weights` and `rng` apply
    as they do outside of this pass. When `None`, only the gates that carry their own
    are drawn.
  * **args** (*Any*)
  * **kwargs** (*Any*)
* **Return type:**
  Any

### Attributes

#### MAX_SAMPLE_RETRIES *= 1000000*

The maximum number of consecutive rejected samples tolerated before [`run()`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.run) gives up and
raises [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError). This guards against an infinite loop when the Hamiltonian’s
remaining terms cannot bridge the tracked occupied/unoccupied mode sets. That happens when both
sets remain small and disjoint (few modes have been marked occupied or unoccupied, and none has
yet become “uncertain”) and no remaining term’s support touches both.

#### qdrift

The fallback synthesis method that draws the terms, or `None` if there is none.

### Methods

#### run(dag)

Runs this transpilation pass.

Each [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) node is replaced by [`num_terms`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift.num_terms) sampled
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates, one per accepted draw. The emitted gates are marked
`Evolution.atomic` and carry over the `Evolution.synthesis` method of the node
they replace. The method that draws them is the node’s own `Evolution.synthesis`
if that is a [`FermionicQDrift`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift), and [`qdrift`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.qdrift) otherwise. A node for which there
is neither is copied to the output unchanged, with a [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning), and every mode it
acts on is then treated as “uncertain” for the filtering of the nodes after it. Nodes that
are not [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates are copied to the output unchanged. Since the sampling is
random, the output varies between runs unless the `rng` was seeded.

* **Parameters:**
  **dag** ([*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)) – the input circuit with fermion-based instructions. Only
  [`DAGOpNode`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode) with [`FermionicGate`](qiskit_fermions.circuit.FermionicGate.html.md#qiskit_fermions.circuit.FermionicGate) instances as their
  [`op`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGOpNode#op) are supported, plus
  `Barrier` and [`FermionicMeasure`](qiskit_fermions.circuit.library.FermionicMeasure.html.md#qiskit_fermions.circuit.library.FermionicMeasure), which are
  carried through untouched (a measurement keeps the classical bit it writes into).
* **Returns:**
  The output circuit which is still acting on a fermionic register. When filtering
  actually ran, its [`metadata`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit#metadata) also carries the
  `filter_trivial.discarded` and `filter_trivial.emitted` counts described in the
  class docstring.
* **Raises:**
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – if [`MAX_SAMPLE_RETRIES`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.MAX_SAMPLE_RETRIES) consecutive draws are rejected without any
        of them coupling the tracked occupied and unoccupied mode sets.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if [`weights`](qiskit_fermions.circuit.library.synthesis.FermionicQDrift.html.md#qiskit_fermions.circuit.library.synthesis.FermionicQDrift.weights) was supplied and its length does not
        match the number of groups (or terms) of an evolved operator.
* **Return type:**
  [*DAGCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)

### Inherited Methods

#### execute(passmanager_ir, state, callback=None)

Execute optimization task for input Qiskit IR.

* **Parameters:**
  * **passmanager_ir** (*IR*) – Qiskit IR to optimize.
  * **state** ([*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)) – State associated with workflow execution by the pass manager itself.
  * **callback** ([*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) *[* *[*[*Task*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.Task) *,* *IR_OUT* *,* [*PropertySet*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PropertySet) *,* [*float*](https://docs.python.org/3/builtins/functions.html#float) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *,* *None* *]*  *|* *None*) – A callback function which is called per execution of optimization task.
* **Returns:**
  Optimized Qiskit IR and state of the workflow.
* **Return type:**
  [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[*IR_OUT*, [*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)]

#### name()

Name of the pass.

* **Return type:**
  [str](https://docs.python.org/3/builtins/stdtypes.html#str)

#### update_status(state, run_state)

Update workflow status.

* **Parameters:**
  * **state** ([*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)) – Pass manager state to update.
  * **run_state** (*RunState*) – Completion status of current task.
* **Returns:**
  Updated pass manager state.
* **Return type:**
  [*PassManagerState*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.PassManagerState)
