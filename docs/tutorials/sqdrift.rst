.. _sqdrift_tutorial:

Generate SqDRIFT Circuits
=========================

`SqDRIFT`_ is a variant of `SQD`_ that replaces the need to choose an ansatz
from which to sample bitstrings with an ensemble of time-evolution circuits
constructed directly from the target Hamiltonian.
This is achieved by subsampling smaller time-evolution operators from said
Hamiltonian based on its coefficients, which is known as the `qDRIFT`_
Trotterization method.

This tutorial shows how to generate an ensemble of such randomized circuits.

1. Hamiltonian Setup
^^^^^^^^^^^^^^^^^^^^

For the purposes of this tutorial, we load the electronic structure Hamiltonian
of N2 from an FCIDUMP file. Of course, there are also other means of
constructing the :class:`.FermionOperator`. Be sure to check out its
documentation as well as :mod:`qiskit_fermions.operators.library`.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.operators.library import FCIDump
       >>> from qiskit_fermions.operators import FermionOperator
       >>>
       >>> fcidump = FCIDump.from_file("docs/tutorials/n2.fcidump")
       >>> hamil = FermionOperator.from_fcidump(fcidump)

    .. code-block:: c

       #include <qiskit_fermions.h>

       QfFCIDump* fcidump = qf_fcidump_from_file("docs/tutorials/n2.fcidump");
       QfFermionOperator* hamil = qf_ferm_op_from_fcidump(fcidump);


2. Group Hamiltonian terms
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this step, we exploit the many symmetries that are present in the electronic
structure Hamiltonian by grouping related terms with identical coefficients.
While doing so changes the operator coefficient distribution which the qDRIFT
protocol samples from, this does not affect its convergence guarantees.
Crucially, the grouping of terms related by symmetry results in a favorable
cancellation of Pauli terms resulting in an overall shorter circuit depth, when
time-evolving a state under their action.

The grouping of terms is straight forward for single excitations (terms of
length 2), while it is less straight forward in the case of double excitations
(terms of length 4). In the loop below, we exploit the fact that we know the
order of operations in our Hamiltonian,
:math:`a^\dagger_{i,\sigma} a^\dagger_{j,\tau} a_{k,\tau} a_{l,\sigma}`,
where :math:`\sigma` and :math:`\tau` indicate the spin species of the fermionic
modes. We can group terms with permuted indices within either species but
(generally speaking) not while mixing the indices across these species.

.. tab-set-code::

    .. code-block:: python

       >>> from collections import defaultdict
       >>>
       >>> groups = defaultdict(FermionOperator.zero)
       >>> ordered_keys = []
       >>> ordered_coeffs = []
       >>>
       >>> for term, coeff in hamil.iter_terms():
       ...     indices = tuple(i for (_, i) in term)
       ...     if len(term) == 2:
       ...         i, j = min(indices), max(indices)
       ...         group_idx = (i, j)
       ...     elif len(term) == 4:
       ...         il, jk = (indices[0], indices[3]), (indices[1], indices[2])
       ...         i, l = min(il), max(il)
       ...         j, k = min(jk), max(jk)
       ...         group_idx = (i, j, k, l)
       ...     elif len(term) == 0:
       ...         # we ignore the identity term
       ...         continue
       ...
       ...     if group_idx not in groups:
       ...         ordered_keys.append(group_idx)
       ...         ordered_coeffs.append(coeff.real)
       ...
       ...     groups[group_idx] += FermionOperator.from_dict({tuple(term): coeff.real})
       >>>
       >>> len(ordered_keys) == len(groups)
       True

    .. code-block:: c

       // TODO!

3. Subsample the Hamiltonian groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step actually performs the `qDRIFT`_ randomization by subsampling the
Hamiltonian groups to produce smaller operators. This step sets `SqDRIFT`_ apart
from `SKQD`_ by ensuring that the time-evolution circuits can still be
implemented efficiently on hardware with limited qubit connectivity, even when
the investigated Hamiltonian is not that of a regular fermionic lattice but
(like here) contains long-range connections and more than quadratic interaction
terms.

This step also introduces the few parameters with which one can tweak the
ensemble of circuits to generate:

* the number of circuits to generate: ``num_circuits``
* the length of each circuit in terms of excitation groups: ``num_exc``
* the factor for the evolution time: ``time``

.. tab-set-code::

    .. code-block:: python

       >>> import numpy as np
       >>>
       >>> time = 1.0
       >>> num_exc = 10
       >>> num_circuits = 100
       >>>
       >>> weights = np.abs(ordered_coeffs)
       >>> lambd = np.sum(weights)
       >>> delta = (lambd * time) / num_exc
       >>>
       >>> rng = np.random.default_rng(42)
       >>> sampled_indices = rng.choice(
       ...     np.arange(len(ordered_coeffs)),
       ...     size=(num_circuits, num_exc),
       ...     p=weights / lambd,
       ... )
       >>>
       >>> subsampled_ops = []
       >>> for sample in sampled_indices:
       ...     op = FermionOperator.zero()
       ...     for idx in sample:
       ...         op += groups[ordered_keys[idx]]
       ...
       ...     subsampled_ops.append(op)

    .. code-block:: c

       // TODO!

4. Map the Operators
^^^^^^^^^^^^^^^^^^^^

In order to implement the time-evolution circuits, we must map the
:class:`.FermionOperator` instances to
:class:`~qiskit.quantum_info.SparseObservable` instances.

In the original proposal of `SqDRIFT`_ the authors performed an additional
optimization to reduce the time-evolution circuit depth by layouting the
fermions on the qubits in different patterns dictated by the excitations
included in each subsampled operator. The effectiveness of this layout
optimization depends on the overlap of the subsampled excitation groups but can
yield drastic improvements.

We do not implement this optimization in this simple example below.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit_fermions.mappers.library import jordan_wigner
       >>>
       >>> norb = fcidump.norb
       >>> num_qubits = 2 * norb
       >>> mapped_ops = [
       ...     jordan_wigner(op, num_qubits).simplify()
       ...     for op in subsampled_ops
       ... ]

    .. code-block:: c

       // TODO!

5. Generate the SqDRIFT circuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we can generate the time-evolution circuits simply with Qiskit's
:class:`~qiskit.circuit.library.PauliEvolutionGate`.

Here, we also prepend each time-evolution with the Hartree Fock bitstring as the
initial state to be evolved. Of course, other initial states may be chosen, too.

.. tab-set-code::

    .. code-block:: python

       >>> from qiskit.circuit import QuantumCircuit
       >>> from qiskit.circuit.library import PauliEvolutionGate
       >>>
       >>> nelec, ms2 = fcidump.nelec, fcidump.ms2
       >>> nelec_a = nelec // 2 + ms2
       >>> nelec_b = nelec - nelec_a
       >>> occupied_a = [1] * nelec_a + [0] * (norb - nelec_a)
       >>> occupied_b = [1] * nelec_b + [0] * (norb - nelec_b)
       >>> initial_state = occupied_a + occupied_b
       >>>
       >>> circuits = []
       >>> for op in mapped_ops:
       ...     circ = QuantumCircuit(num_qubits)
       ...     _ = circ.x(initial_state)
       ...     _ = circ.append(PauliEvolutionGate(op, delta), circ.qubits)
       ...
       ...     circuits.append(circ)

    .. code-block:: c

       // WARNING: Qiskit's C API does not yet support the PauliEvolutionGate.
       // This feature is planned to be released in Qiskit 2.4

Next steps
^^^^^^^^^^

Now that we have successfully generated an ensemble of circuits, we must sample
bitstrings from them. To do so, the circuits must be transpiled and subsequently
sent to hardware for execution. We will not cover this here, and instead refer
to the `Qiskit documentation
<https://quantum.cloud.ibm.com/docs/en/guides/intro-to-patterns>`_ for detailed
guides on the various steps involved.

Once the bitstring samples have been obtained, these can be used in combination
with the Hamiltonian coefficients to perform the SQD post-processing, a great
guide for which is written up in the `SQD addon tutorials
<https://qiskit.github.io/qiskit-addon-sqd/tutorials/index.html>`_.


.. _qDRIFT: https://arxiv.org/abs/1811.08017
.. _SQD: https://arxiv.org/abs/2405.05068
.. _SqDRIFT: https://arxiv.org/abs/2508.02578
.. _SKQD: https://arxiv.org/abs/2501.09702
