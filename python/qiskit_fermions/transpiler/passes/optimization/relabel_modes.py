# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An optimization pass to relabel fermionic modes."""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Generator, Iterable
from itertools import islice
from typing import TYPE_CHECKING, Any

from qiskit import QuantumRegister

from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
from qiskit_fermions.circuit import FermionicDAGCircuit, FermionicRegister
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.optimization import build_excitation_span_minimization_model
from qiskit_fermions.utils.optionals import HAS_PYOMO

from ... import FermionicDAGCircuitPass

if TYPE_CHECKING:
    import pyomo


def _sliding_window(iterable: Iterable[Any], n: int) -> Generator[tuple[Any, ...], None, None]:
    """Collect data into overlapping fixed-length chunks or blocks.

    Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes

    Example: sliding_window('ABCDEFG', 3) → ABC BCD CDE DEF EFG
    """
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


class RelabelModes(FermionicDAGCircuitPass):
    """A transpilation pass to relabel the fermionic modes.

    .. rubric:: Post-processing

    The :class:`.FermionicDAGCircuit` returned by this transpiler pass will have a new field in its
    :attr:`~qiskit.dagcircuit.DAGCircuit.metadata` called ``permutation`` which will contain the
    fermionic mode index re-labeling that was applied by this transpiler pass.

    .. note::
       This metadata may **not** be equal to the value of :attr:`permutation`, specifically when the
       automatic permutation optimization gets used.

    When working with this transpiler pass, bitstrings sampled from the final circuit will need to
    have their bits re-ordered according to the reverse permutation:

    .. doctest::

       >>> from qiskit.passmanager import MultiStagePassManager
       >>> from qiskit.providers.basic_provider import BasicSimulator
       >>> from qiskit_fermions.circuit import FermionicCircuit
       >>> from qiskit_fermions.circuit.library import InitializeModes
       >>> from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
       >>> from qiskit_fermions.transpiler.passes import (
       ...     F2QSynthesis, F2QSynthesisPluginManager, RelabelModes, TrivialF2QLayout,
       ... )
       >>>
       >>> circ = FermionicCircuit(4)
       >>> circ.append(InitializeModes([1, 1, 0, 0]), circ.modes)
       >>>
       >>> synth_plugins = F2QSynthesisPluginManager()
       >>> synth = F2QSynthesis()
       >>> synth.methods["InitializeModes"] = synth_plugins.method("InitializeModes", "TrivialOccupation")()
       >>>
       >>> relabel = RelabelModes(permutation=[0, 2, 1, 3])
       >>>
       >>> pm = MultiStagePassManager(
       ...     init=FermionicCircuitToDAG(),
       ...     optimization=relabel,
       ...     layout=TrivialF2QLayout(),
       ...     synthesis=synth,
       ...     output=QuantumDAGToCircuit(),
       ... )
       >>>
       >>> qcirc = pm.run(circ)
       >>> qcirc.measure_all()
       >>>
       >>> bit_permutation = qcirc.metadata["permutation"]
       >>> print(bit_permutation)
       [0, 2, 1, 3]
       >>>
       >>> res = BasicSimulator().run(qcirc, shots=1).result()
       >>> counts = res.get_counts()
       >>>
       >>> # undo bit permutation (note the negative idx due to the small-endian convention of
       >>> # Qiskit's qubit ordering)
       >>> post_processed = {
       ...     "".join(bitstring[-idx] for idx in bit_permutation): count
       ...     for bitstring, count in counts.items()
       ... }
       >>> print(post_processed)
       {'0011': 1}
    """

    def __init__(
        self,
        permutation: list[int] | None = None,
        *,
        solver: pyomo.opt.SolverFactory | None = None,
        **kwargs,
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            permutation: the index permutation used to relabel the fermionic mode indices. When this
                is ``None``, a permutation will be determined automatically based on
                :func:`.build_excitation_span_minimization_model`. See also :attr:`.permutation` for
                more details.
            solver: the optimization problem solver instance used to solve the
                :func:`.build_excitation_span_minimization_model` problem. When this is ``None``, no
                ``permutation`` can be determined automatically. See also :attr:`.solver` for more
                details.
            kwargs: any additional keyword arguments will be forward to
                :func:`.build_excitation_span_minimization_model`.
        """
        super().__init__()

        self.permutation = permutation
        """The index permutation used to relabel the fermionic mode indices.

        This may either be a ``list[int]`` in which case its length has to match the number of
        fermionic modes of the circuit being transpiled. This scenario therefore requires the
        transpiler pass to be tailored quite specifically to the user's circuit.

        Being a permutation, each index in this list has to appear exactly once.

        Or it may be ``None``, in which case the :func:`.build_excitation_span_minimization_model`
        function is used to define an optimization problem which tries to minimize the span of all
        occurring fermionic excitations.

        .. note::
           The use of this optimization model is only implemented for time evolution gates
           containing a :class:`.FermionOperator` instance.
        """

        self.solver = solver
        """The optimization problem solver instance to automatically find :attr:`.permutation`.

        When :attr:`.permutation` is ``None``, the optimization problem defined by
        :func:`.build_excitation_span_minimization_model` is used to automatically find a good
        permutation of mode indices. In such a case, the user must provide an optimizer to solve
        this model.
        """

        self._model_kwargs = kwargs

    def find_permutation(
        self, dag: FermionicDAGCircuit
    ) -> tuple[list[int] | None, pyomo.opt.results.results_.SolverResults | None]:
        """Finds a mode index :attr:`.permutation` when not specified by the user.

        This function only gets called when :attr:`.permutation` is not specified by the user (i.e.
        it is ``None``). When that is the case, it does the following:

        1. ensure that the optional `pyomo <https://pypi.org/project/pyomo/>`_ dependency is
           installed. Otherwise, no optimization can be performed and this transpiler pass has no
           effect.
        2. ensure that a :attr:`.solver` is specified. Otherwise, no optimization can be performed
           and this transpiler pass has no effect.
        3. gather all the fermionic excitations from any :class:`.Evolution` gates containing a
           :class:`.FermionOperator` instance.
        4. build the optimization problem using :func:`.build_excitation_span_minimization_model`,
           forwarding any additional keyword arguments (``kwargs``) from when this transpiler pass
           was constructed.
        5. solve the optimization problem using :attr:`.solver` and extract the final permutation.

        Args:
            dag: the circuit to be transpiled.

        Returns:
            The permutation to use. When ``None``, this transpiler pass will have no effect.

        Raises:
            NotImplementedError: when encountering an :class:`.Evolution` gate containing an
                operator that is not a :class:`.FermionOperator` instance.
        """
        if not HAS_PYOMO:
            warnings.warn(
                "Finding the optimal mode index permutation requires the optional 'pyomo' "
                "dependency to be installed.",
                category=UserWarning,
                stacklevel=1,
            )
            return (None, None)

        if self.solver is None:
            warnings.warn(
                "Finding the optimal mode index permutation requires the `RelabelModes.solver` "
                "attribute to be defined.",
                category=UserWarning,
                stacklevel=1,
            )
            return (None, None)

        gathered_excitations: list[tuple[int, ...]] = []
        for node in dag.op_nodes():
            if not isinstance(node.op, Evolution):
                continue

            hamil = node.op.operator
            if not isinstance(hamil, FermionOperator):
                raise NotImplementedError(
                    "The optimization model defined by build_excitation_span_minimization_model "
                    "assumes fermionic mode excitations defined in terms of `FermionOperator` "
                    "instances. Handling time evolution gates of operators of type {} is not "
                    "implemented.",
                    type(hamil),
                )

            modes = hamil.get_modes()
            boundaries = hamil.get_boundaries()
            for start, stop in _sliding_window(boundaries, 2):
                gathered_excitations.append(tuple(modes[start:stop]))

        num_modes = dag.num_qubits()
        model = build_excitation_span_minimization_model(
            gathered_excitations, num_modes, **self._model_kwargs
        )

        try:
            result = self.solver.solve(model)
        except RuntimeError as exc:
            warnings.warn(
                f"Encountered a RuntimeError while trying to optimize the permutation: {exc}",
                category=UserWarning,
                stacklevel=1,
            )
            return (None, None)

        from pyomo.environ import value

        permutation = [round(value(model.y[i])) for i in range(num_modes)]
        return permutation, result

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit which is still acting on a fermionic register.

        Raises:
            NotImplementedError: when the provided input circuit has more than a single register.
        """
        if len(dag.qregs) > 1:
            raise NotImplementedError(
                "Cannot apply the RelabelModes pass to a circuit with more than a single register."
            )

        permutation, opt_result = (
            self.find_permutation(dag) if self.permutation is None else (self.permutation, None)
        )

        if permutation is None:
            return dag

        out_dag = dag.copy_empty_like()
        out_dag.metadata["permutation"] = permutation.copy()
        if opt_result is not None:
            out_dag.metadata["permutation.opt_result"] = opt_result

        # NOTE: we already know out_dag.qregs to contain only a single entry
        for orig_name, orig_register in out_dag.qregs.items():
            out_dag.remove_qregs(orig_register)
            relabeled_register: FermionicRegister = QuantumRegister(
                name=f"{orig_name}'", bits=[orig_register[idx] for idx in permutation]
            )
            out_dag.add_qreg(relabeled_register)

        for node in dag.op_nodes():
            orig_indices = [orig_register.index(mode) for mode in node.qargs]
            out_dag.apply_operation_back(
                node.op, qargs=[relabeled_register[idx] for idx in orig_indices]
            )

        return out_dag
