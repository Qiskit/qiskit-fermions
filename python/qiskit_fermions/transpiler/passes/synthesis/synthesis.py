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

"""Fermion-to-qubit circuit synthesis pass."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

from qiskit.circuit import Barrier
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import GenericPass

from qiskit_fermions.circuit import FermionicDAGCircuit, FermionicGate

from ... import F2QLayout
from .plugin import F2QSynthesisPlugin, F2QSynthesisPluginManager

F2QSynthesisConfig: TypeAlias = dict[
    str, str | tuple[str, Sequence[Any]] | tuple[str, Sequence[Any], Mapping[str, Any]]
]
"""The dictionary type used to configure the :attr:`.F2QSynthesis.methods`.

The keys of this dictionary must be names of :class:`.FermionicGate` circuit instructions.
The values can be one of three types:

1. ``str``: the simplest scenario simply specifies the name of the plugin method to use for the
   corresponding key in this dictionary. The plugin is extracted from the
   :class:`.F2QSynthesisPluginManager` and it's ``__init__`` method may **not** require any
   arguments.
2. ``tuple[str, Sequence[Any]]``: the first ``str`` is the same as the above, and the second
   ``Sequence[Any]`` can be used to provide any positional arguments to the plugin's ``__init__``
   method.
3. ``tuple[str, Sequence[Any], Mapping[str, Any]]``: the first two fields are the same as in 2. and
   the third ``Mapping[str, Any]`` can be used to provide any keyword arguments to the plugin's
   ``__init__`` method.
"""


class F2QSynthesis(GenericPass[FermionicDAGCircuit, DAGCircuit]):
    """A transpilation pass to map fermion-based circuit instructions to qubit-based ones.

    This transpilation pass works similarly to Qiskit's
    :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass; given an input
    :class:`.FermionicDAGCircuit`, it iterates the contained instructions and delegates the
    translation to qubit-based instructions to matching :attr:`methods`. The insertion of the
    qubit-based circuit instructions into the output :class:`~qiskit.dagcircuit.DAGCircuit` is also
    left to the plugin method. This pass will merely have prepared the
    :class:`~qiskit.circuit.QuantumRegister` according to the global transpilation
    :class:`~qiskit_fermions.transpiler.F2QLayout` setting.

    .. rubric:: Usage

    There are two ways to configure the plugins used by this transpiler pass. The examples below
    show the equivalent configuration to the :func:`.generate_preset_jw_pass_manager`.

    1. Providing a ``config`` during initialization:

       .. doctest::

          >>> from qiskit_fermions.mappers.library import jordan_wigner
          >>> from qiskit_fermions.transpiler import passes
          >>>
          >>> config = {
          ...     "Evolution": ("MapperFn", (jordan_wigner,)),
          ...     "InitializeModes": "TrivialOccupation",
          ...     "OrbitalRotation": "GivensDecomposition",
          ... }
          >>> synth = passes.F2QSynthesis(config)

    2. Manually populating the :attr:`methods` attribute:

       .. doctest::

          >>> from qiskit_fermions.mappers.library import jordan_wigner
          >>> from qiskit_fermions.transpiler import passes
          >>>
          >>> synth = passes.F2QSynthesis()
          >>> synth.methods["Evolution"] = passes.MapperFnEvolutionSynthesis(jordan_wigner)
          >>> synth.methods["InitializeModes"] = passes.TrivialOccupationInitializeModesSynthesis()
          >>> synth.methods["OrbitalRotation"] = passes.GivensDecompositionOrbitalRotationSynthesis()

       Through this manual approach it is also possible to inject custom
       :class:`.F2QSynthesisPlugin` implementations without requiring them to be registered through
       an `entry-point <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_.

    """

    def __init__(self, config: F2QSynthesisConfig | None = None) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            config: an optional dictionary to pre-populate the :attr:`methods` with plugins provided
                by the :class:`.F2QSynthesisPluginManager`.
        """
        super().__init__()

        self.methods: dict[str, F2QSynthesisPlugin] = {}
        """A dictionary of fermion-to-qubit circuit instruction transpilation methods.

        For this transpilation pass to have any effect, this dictionary must be populated with
        instances of the :class:`.F2QSynthesisPlugin` protocol. The keys of this dictionary
        correspond to the ``__name__`` of a circuit instruction.

        All available transpilation plugins are managed by the :class:`.F2QSynthesisPluginManager`.

        .. note::
           This dictionary is **empty** by default! You can pre-populate it with plugins provided by
           the :class:`.F2QSynthesisPluginManager` by providing a ``config`` argument during
           initialization of this transpiler pass instance.
        """

        if config is not None:
            pm = F2QSynthesisPluginManager()
            for op_name, method_spec in config.items():
                init_args: Sequence[Any] = ()
                init_kwargs: Mapping[str, Any] = {}
                match method_spec:
                    case str():
                        method_name = method_spec
                    case (name, args):
                        method_name = name
                        init_args = args
                    case (name, args, kwargs):
                        method_name = name
                        init_args = args
                        init_kwargs = kwargs

                method = pm.method(op_name, method_name)
                self.methods[op_name] = method(*init_args, **init_kwargs)

    def run(self, dag: FermionicDAGCircuit) -> DAGCircuit:
        """Runs this transpilation pass.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported, plus
                :class:`~qiskit.circuit.library.Barrier`, which is mapped onto every qubit of each
                register it touched.

        Returns:
            The output circuit with qubit-based instructions.

        Raises:
            ValueError: when a :class:`~qiskit.dagcircuit.DAGOpNode` is encountered whose
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` is neither of type :class:`.FermionicGate`
                nor a :class:`~qiskit.circuit.library.Barrier`.
            TypeError: when a :class:`.FermionicGate` type is encountered for which no translation
                plugin is present in :attr:`methods`.
        """
        f2q_layout = cast(F2QLayout, self.property_set["f2q_layout"])

        # Build the qubit-based output DAG from scratch rather than copying the fermionic input and
        # deleting its mode wires: ``copy_empty_like`` followed by ``remove_qregs``/``remove_qubits``
        # leaves the DAG's internal wire graph in a state that intermittently trips a spurious
        # ``not a DAG`` panic in downstream analysis (e.g. Qiskit's ``Depth`` pass). Starting fresh
        # and only adding the mapped qubit registers keeps the graph consistent.
        out_dag = DAGCircuit()
        out_dag.name = dag.name
        out_dag.global_phase = dag.global_phase
        out_dag.metadata = dag.metadata
        for qreg in f2q_layout.values():
            out_dag.add_qreg(qreg)
        # Copy the clbits *before* the registers: this DAG is built from scratch, so anything not
        # copied explicitly is absent, and a clbit outside any register would otherwise be dropped --
        # emitting an operation onto it then fails with an opaque "not present" KeyError. Adding them
        # in the input's own order also pins the clbit indices, which are what the positions of a
        # sampled bitstring mean.
        out_dag.add_clbits(dag.clbits)
        for creg in dag.cregs.values():
            out_dag.add_creg(creg)

        for node in dag.op_nodes():
            # Key on `base_class`, not `type(...)`: a singleton instruction (such as
            # `FermionicMeasure`, which subclasses Qiskit's `Measure` so the drawers render it) is an
            # instance of a generated `_Singleton...` subclass, whose name no entry point could match.
            # `base_class` is the user-facing class, and is identical to `type(...)` for every
            # non-singleton instruction, so this is the right key in both cases.
            op_type = node.op.base_class

            # A barrier is not a FermionicGate but is supported: it carries no fermionic content, so it
            # needs no synthesis plugin, only a placement on the mapped qubits. Because an F2QLayout
            # relates whole registers (whose two sides can even differ in size), the modes a fermionic
            # barrier covers have no per-qubit image in general, so span every qubit of each mapped
            # register it touched. That is correct under any layout, and a barrier can span several
            # registers, which is why this does not go through `map_node_single_register`.
            if isinstance(node.op, Barrier):
                qargs = [
                    qubit
                    for freg, qreg in f2q_layout.items()
                    if any(mode in freg for mode in node.qargs)
                    for qubit in qreg
                ]
                out_dag.apply_operation_back(Barrier(len(qargs), label=node.op.label), qargs=qargs)
                continue

            if not isinstance(node.op, FermionicGate):
                raise ValueError("Encountered an unsupported circuit instruction type: {}", op_type)

            plugin = self.methods.get(op_type.__name__, None)
            if plugin is None:
                raise TypeError(
                    "No plugin registered for transpiling a circuit instruction of type: {}",
                    op_type,
                )

            plugin.run(node, out_dag, f2q_layout=f2q_layout)

        return out_dag
