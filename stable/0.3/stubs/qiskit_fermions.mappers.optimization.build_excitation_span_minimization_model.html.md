# build_excitation_span_minimization_model

### build_excitation_span_minimization_model(excitations, num_modes, , objective='multi', mix_delta=0.1)

Build a Pyomo model for ordering fermionic modes to minimize excitation spans.

The model constructs a permutation where each original mode index is assigned to exactly one
position in a linear ordering. For every 2-mode and 4-mode excitation tuple, the span of
occupied positions is minimized according to the chosen objective.

#### NOTE
The `excitations` list is pre-processed as follows:

- indices that occur twice within the same tuple get cancelled
- resulting tuples of length 0 or 1 are ignored, therefore not imposing a distance constraint
  in the resulting minimization model
- consequently, only post-processed tuples of length 2 or 4 are expected

#### SEE ALSO
For a more detailed discussion and explanation of the reasoning behind this pre-processing,
refer to section III.B of [this paper](https://arxiv.org/abs/2508.02578v2).

* **Parameters:**
  * **excitations** (*Sequence* *[*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]*  *|* [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *]*) – a sequence of excitation index tuples over fermionic mode indices.
  * **num_modes** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – the total number of fermionic modes to be ordered.
  * **objective** (*Literal* *[* *'minmax'* *,*  *'multi'* *,*  *'avg'* *]*) – 

    the chosen objective mode. This may be one of the following literals:
    - `minmax`: minimizes the maximum excitation span.
    - `avg`: minimizes the average excitation span.
    - `multi`: minimizes `max_span + mix_delta * average_span`.
  * **mix_delta** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – the mixing weight used in the case of `objective="multi"`.
* **Returns:**
  A Pyomo optimization model encoding the permutation variables, span variables, constraints,
  and objective. Solve this model with a Pyomo solver to obtain an optimized ordering.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if a post-processed excitation tuple has length unequal to 2 or 4.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if an unknown `objective` is chosen.
* **Return type:**
  [pyomo.core.base.PyomoModel.ConcreteModel](https://pyomo.readthedocs.io/en/stable/api/pyomo.core.base.PyomoModel.ConcreteModel.html#pyomo.core.base.PyomoModel.ConcreteModel)
