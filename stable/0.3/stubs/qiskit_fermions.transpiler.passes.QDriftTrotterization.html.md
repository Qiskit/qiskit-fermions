# QDriftTrotterization

### *class* QDriftTrotterization(num_terms, , filter_trivial=False, weights=None, rng=None)

A transpilation pass to Trotterize [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates via the qDRIFT protocol.

Bases: [`GenericPass`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.passmanager.GenericPass)[[`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit), [`DAGCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit)]

This pass replaces the exact evolution $e^{-i t H}$ of each [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate by a
randomized product formula: it draws `num_terms` samples from the Hamiltonian’s terms (or
[`groups`](qiskit_fermions.operators.FermionOperator.html.md#qiskit_fermions.operators.FermionOperator.groups), if assigned), with each term sampled
with a probability proportional to the magnitude of its sampling weight $w_j$, and emits
one [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate per sample. Every sampled gate evolves its (unit-magnitude,
sign-preserving) term (or, when the Hamiltonian carries groups, its whole sampled group) for
the same time

$$
\delta = \frac{\lambda t}{\texttt{num\_terms}}, \qquad
p_j = \frac{w_j}{\lambda}, \qquad
\lambda = \sum_j w_j,
$$

so that each draw contributes $\delta \cdot p_j = w_j t / \texttt{num\_terms}$. By default
$w_j = |c_j|$, the magnitude of the sampled term’s (or group’s) coefficient, which recovers
the textbook qDRIFT normalization $\lambda = \sum_j |c_j|$; the sign of $c_j$ is not
part of the weight but of the sampled operator, and is read off it directly. Supply
[`weights`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.weights) to precompute that array once instead of deriving it on every call.

The ordered product of the sampled evolutions does not reproduce $e^{-i t H}$ exactly;
rather, its expectation over the sampling approximates the exact evolution, with an error that
decreases as `num_terms` grows. Because the output depends on the random draws, it differs
from run to run unless a fixed `rng` is supplied.

#### NOTE
The sampled gates are marked [`Evolution.atomic`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.atomic): the random draw *is* the
Trotterization this pass performs, so the emitted gates are terminal factors and
[`decompose()`](qiskit_fermions.circuit.FermionicCircuit.html.md#qiskit_fermions.circuit.FermionicCircuit.decompose) leaves them in place rather than splitting them further.
Each of them carries over the [`Evolution.synthesis`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.synthesis) method of the gate it was sampled
from, even though an atomic gate never consults it.

#### NOTE
A fixed `rng` reproduces the *sequence* of randomizations, not an individual member of it.
Successive [`run()`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.run) calls draw from the same generator, so they yield different circuits;
replaying from the same seed reproduces all of them in the same order, but the nth circuit
cannot be obtained without drawing the preceding ones first. Generating a batch of
randomizations and recording the seed therefore works as expected (see
[Generate SqDRIFT circuits](../guides/sqdrift.html.md#sqdrift-getting-started)); addressing one member directly is not supported.

#### WARNING
The optional `filter_trivial` mode exists to stop the sampling from spending one of the
`num_terms` slots on an excitation that cannot move a particle, and which therefore tells
you nothing about the sampled bitstrings. It buys that with the protocol’s convergence
guarantee, and it is off by default for that reason.

It does not make the circuit shorter. `num_terms` is fixed, so a rejected draw is replaced
rather than dropped, and the cheap term it would have contributed (a diagonal rotation, say)
gives way to a coupling excitation that costs more to synthesize. Expect the filtered circuit
to be deeper than the unfiltered one: the budget of sampled slots is what the filtering
conserves, not the depth.

Rejecting a drawn term and re-drawing renormalizes the sampling distribution over the
accepted terms only, so the sampled product no longer averages to $e^{-i t H}$ for the
Hamiltonian you passed in: the effective coefficient of every retained term is inflated by
the reciprocal of the acceptance probability, and the rejected terms drop out. Neither
$\lambda$ nor $\delta$ is adjusted to compensate, so the distortion does not
cancel.

Use it only when the sampled bitstrings are the quantity of interest, as in the SqDRIFT
workflow of [Generate SqDRIFT circuits](../guides/sqdrift.html.md#sqdrift-getting-started). Do not use it when the sampled circuits are used
to estimate an expectation value, a time-evolved observable, or anything else that relies on
the qDRIFT error bound: those results are biased by an amount the pass does not track. See
[`filter_trivial`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.filter_trivial) for the acceptance rule and its prerequisites.

#### HINT
Terms that are diagonal in the occupation-number basis (that is, products of number operators)
have no effect on the sampled bitstrings, so including them only increases the sampling
overhead. Filter them out with
[`filter_diagonal_terms()`](qiskit_fermions.operators.terms.filter_diagonal_terms.html.md#qiskit_fermions.operators.terms.filter_diagonal_terms) on the Hamiltonian
*before* constructing the [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate, rather than on every call to [`run()`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.run):
this pass runs once per transpiled circuit, so filtering upstream avoids repeating the same
filtering work for every circuit generated from the same Hamiltonian.

#### CAUTION
The *scale* of [`weights`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.weights) is not free: it sets the evolution time. Since
$\delta \cdot p_j = w_j t / \texttt{num\_terms}$ above depends on $w_j$ itself and
not merely on its share of $\lambda$, rescaling every weight by $\gamma$ leaves the
distribution untouched but evolves for $\gamma t$ rather than $t$. Weights are
therefore absolute magnitudes, not relative preferences, and only $w_j = |c_j|$
reproduces $e^{-i t H}$.

A distribution whose *shape* differs from $|c_j|$ no longer approximates
$e^{-i t H}$ on its own either. Recovering the target evolution then requires
reweighting the measured outcomes, which is the caller’s responsibility: this pass emits
circuits and cannot post-process their results. See [`weights`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.weights).

#### SEE ALSO
The qDRIFT protocol was introduced in [arXiv:1811.08017](https://arxiv.org/abs/1811.08017).

### Filtering diagnostics

When `filter_trivial` actually filters a gate, the returned [`FermionicDAGCircuit`](qiskit_fermions.circuit.FermionicDAGCircuit.html.md#qiskit_fermions.circuit.FermionicDAGCircuit)
records how many draws it discarded in its [`metadata`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.dagcircuit.DAGCircuit#metadata), under
`filter_trivial.discarded` and `filter_trivial.emitted`. Both hold one entry per filtered
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate, in circuit order. Their ratio estimates the acceptance probability,
and hence the factor by which the filtering inflated the coefficients of the terms it kept: an
acceptance probability close to one means the filtering barely moved the distribution, while a
small one means the retained terms were weighted far above their true share of the Hamiltonian.

#### IMPORTANT
Neither field is present when no gate was filtered, which covers `filter_trivial=False` and
every case in which filtering was skipped with a [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning). Read them defensively,
for example with `qcirc.metadata.get("filter_trivial.discarded")`. A discarded count of
zero is different from an absent field: it says the filtering ran on that gate and accepted
every draw, so it left the sampling distribution untouched.

Initializing this transpiler pass can be done with the arguments listed below.

* **Parameters:**
  * **num_terms** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the number of terms to sample for the qDRIFT Trotterization. This equals the
    number of [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates emitted per input gate; a larger value reduces
    the Trotterization error at the cost of a deeper circuit.
  * **filter_trivial** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – when set to `True`, the sampling loop rejects a drawn term unless its
    support couples a mode tracked as occupied with a mode tracked as unoccupied, and
    draws a replacement in its place. This spends every one of the `num_terms`
    slots on an excitation that can move a particle, at the cost of biasing the
    Trotterization and of a deeper circuit, so it defaults to `False`. See also
    [`filter_trivial`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.filter_trivial) for the acceptance rule and its prerequisites, and the
    warning in the class docstring for the bias it introduces.
  * **weights** (*Sequence* *[*[*float*](https://docs.python.org/3/builtins/functions.html#float) *]*  *|* *np.ndarray* *|* *None*) – the sampling weights to use instead of the coefficient magnitudes derived from
    the Hamiltonian. If `None` (the default), they are computed from the evolved
    operator on every call, which reproduces the textbook qDRIFT distribution. See
    [`weights`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.weights) for the expected length, the sign convention and the effect on the
    evolution time.
  * **rng** (*np.random.Generator* *|* [*int*](https://docs.python.org/3/builtins/functions.html#int) *|* *None*) – the random number generator (rng) to be used. When this is an `int`, the internal
    rng will be initialized with `np.random.default_rng(seed=rng)`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `weights` is not one-dimensional, is empty, holds a non-finite or
      negative entry, or sums to zero.

### Attributes

#### MAX_SAMPLE_RETRIES *= 1000000*

The maximum number of consecutive rejected samples tolerated by `filter_trivial` before
[`run()`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.run) gives up and raises [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError). This guards against an infinite loop when
the Hamiltonian’s remaining terms cannot bridge the tracked occupied/unoccupied mode sets. That
happens when both sets remain small and disjoint (few modes have been marked occupied or
unoccupied, and none has yet become “uncertain”) and no remaining term’s support touches
both.

#### num_terms

The number of terms to include in the qDRIFT Trotterization.

#### filter_trivial

Whether to reject drawn terms that do not couple the tracked occupied/unoccupied modes.

When this is `True`, the sampling loop accepts a drawn term only if its support intersects
both the set of modes tracked as occupied and the set tracked as unoccupied, and it draws a
replacement for every term it rejects, so that none of the [`num_terms`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.num_terms) slots is spent
on a term that leaves the occupation unchanged and therefore says nothing about the sampled
bitstrings. Rejection renormalizes the sampling distribution over the accepted terms, which
biases the Trotterization, and the replacement it draws is more expensive to synthesize than
the term it displaced: see the warning in the class docstring before enabling this.

Filtering requires an [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) or [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) gate
to precede the [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates being Trotterized, to seed the initial occupied and
unoccupied mode sets. If none is found, or if the sets it seeds turn out to be entirely
occupied or entirely unoccupied, filtering is skipped for that gate and a
[`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning) is emitted instead.

Any [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) gate encountered before or between the [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)
gates also updates these sets: every mode it acts on becomes “uncertain”, since the rotation
can mix it with any other mode it touches, just like a mode touched by an accepted term. A
[`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) gate updates the sets the same way its
[`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) and [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) components would if applied in
sequence: it seeds the occupied and unoccupied sets from its `occupation`, then
immediately marks every mode it acts on as “uncertain” because of its rotation. See the
[`run()`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.run) docstring for the precise acceptance rule.

#### weights *: np.ndarray | [None](https://docs.python.org/3/builtins/constants.html#None)*

The sampling weights $w_j$, or `None` to derive them from the evolved operator.

Supplying them hoists their computation out of the transpilation: the default path recomputes
them on *every* call to [`run()`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.run), which is repeated work when generating an ensemble from a
single Hamiltonian. Deriving them once with
[`group_coeff_means()`](qiskit_fermions.operators.terms.group_coeff_means.html.md#qiskit_fermions.operators.terms.group_coeff_means) and passing the result here keeps
this pass stateless while paying that cost a single time. That is the reason to reach for this
argument; passing anything other than the Hamiltonian’s own coefficient magnitudes changes
which evolution the ensemble approximates (see the caution below).

One entry is expected per *group* when the evolved operator carries
[`groups`](qiskit_fermions.operators.OperatorTrait.html.md#qiskit_fermions.operators.OperatorTrait.groups), and per *term* otherwise; a
mismatch raises [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError). Because such an array describes one specific operator,
a circuit holding more than one [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate is rejected as well: leave this
unset for such a circuit, so that every gate derives its own weights, or transpile one gate
at a time.

The granularity follows what the pass samples, which is why it is the grouping that decides
it: a grouped operator is sampled group-wise, so a weight describes a whole group. Whether
the grouping is the appropriate unit for a given operator is a property of that operator, not
of this argument – see [Group operator terms: use the operator structure](../guides/grouping.html.md#grouping-explanation), and
[`groups_are_hermitian()`](qiskit_fermions.operators.terms.groups_are_hermitian.html.md#qiskit_fermions.operators.terms.groups_are_hermitian) to check the most
common convention.

Entries must be non-negative. A weight is the magnitude $h_j$ of the qDRIFT
decomposition $H = \sum_j h_j H_j$, in which a coefficient’s sign belongs to
$H_j$ rather than to $h_j$ and is read off the operator directly, so a sign here
would have nothing to describe. Sampling from a signed (quasi-probability) distribution is a
separate feature: it additionally requires the accumulated sign of the sampled entries to be
reported back for post-processing, which this pass does not do.

#### CAUTION
These are absolute magnitudes, not relative preferences: their sum also sets the evolution
time, so rescaling every entry by $\gamma$ evolves for $\gamma t$ while
sampling identically. See the class docstring.

### Methods

#### run(dag)

Runs this transpilation pass.

Each [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) node is replaced by `num_terms` sampled [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution)
gates, one per drawn term (or per drawn group, when the Hamiltonian carries groups; see the
class docstring). The emitted gates are marked [`Evolution.atomic`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.atomic) and carry over the
[`Evolution.synthesis`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution.synthesis) method of the node they replace. Nodes that are not
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates are copied to the output unchanged. Since the sampling is random,
the output varies between runs unless the `rng` was seeded.

The sampling weights are recomputed from each evolved operator here, unless [`weights`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.weights)
was supplied, in which case that array is used as-is and this method never touches the
operator’s coefficients. A supplied array is validated against the operator it is applied to,
and restricts the circuit to a single [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate (see [`weights`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.weights)).

When [`filter_trivial`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.filter_trivial) is set, this method tracks the sets of modes that are known to
be occupied or unoccupied, seeded from any [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) gate(s) preceding the
[`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gates in the circuit (several such gates placed in parallel, for example
one per spin sector, are accumulated together). A drawn term is accepted only if its support
intersects *both* sets, that is, it couples a known-occupied mode with a known-unoccupied
one; otherwise it is discarded and a replacement is drawn. Rejection renormalizes the
sampling distribution over the accepted terms, which biases the Trotterization: see the
warning in the class docstring. Once a term is accepted, every mode in its support becomes
“uncertain” and is added to *both* sets, making it eligible to participate in either role
for subsequent samples. Any [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) gate found in the circuit updates
these sets the same way: every mode it acts on becomes “uncertain” too, since the rotation
can mix it with any other mode in its support. A [`PrepareSlaterDeterminant`](qiskit_fermions.circuit.library.PrepareSlaterDeterminant.html.md#qiskit_fermions.circuit.library.PrepareSlaterDeterminant) gate is
treated as its [`InitializeModes`](qiskit_fermions.circuit.library.InitializeModes.html.md#qiskit_fermions.circuit.library.InitializeModes) and [`OrbitalRotation`](qiskit_fermions.circuit.library.OrbitalRotation.html.md#qiskit_fermions.circuit.library.OrbitalRotation) components applied
back-to-back: its `occupation` first seeds the occupied/unoccupied sets, and then every
mode it acts on is immediately marked “uncertain”, since it also carries a rotation.

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
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – if `filter_trivial` is `True` and [`MAX_SAMPLE_RETRIES`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.MAX_SAMPLE_RETRIES)
        consecutive draws are rejected without any of them coupling the tracked occupied and
        unoccupied mode sets.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if [`weights`](#qiskit_fermions.transpiler.passes.QDriftTrotterization.weights) was supplied and its length does not match the number of
        groups (or terms) of an evolved operator, or if the circuit holds more than one
        [`Evolution`](qiskit_fermions.circuit.library.Evolution.html.md#qiskit_fermions.circuit.library.Evolution) gate.
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
