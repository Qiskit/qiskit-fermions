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

"""Fermion-to-fermion gate synthesis pass."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from qiskit_fermions.circuit import FermionicDAGCircuit, FermionicGate
from qiskit_fermions.circuit.library.synthesis import F2FSynthesisPlugin, F2FSynthesisPluginManager

from ... import FermionicDAGCircuitPass

F2FSynthesisConfig: TypeAlias = dict[
    str,
    str
    | tuple[str, Sequence[Any]]
    | tuple[str, Sequence[Any], Mapping[str, Any]]
    | F2FSynthesisPlugin,
]
"""The dictionary type used to configure the :attr:`.F2FSynthesis.methods`.

The keys of this dictionary must be names of :class:`.FermionicGate` classes that can be broken down
in fermionic space. The values can be one of four types:

1. ``str``: the simplest scenario simply specifies the name of the plugin method to use for the
   corresponding key in this dictionary. The plugin is extracted from the
   :class:`.F2FSynthesisPluginManager` and its ``__init__`` method may **not** require any
   arguments.
2. ``tuple[str, Sequence[Any]]``: the first ``str`` is the same as the above, and the second
   ``Sequence[Any]`` can be used to provide any positional arguments to the plugin's ``__init__``
   method.
3. ``tuple[str, Sequence[Any], Mapping[str, Any]]``: the first two fields are the same as in 2. and
   the third ``Mapping[str, Any]`` can be used to provide any keyword arguments to the plugin's
   ``__init__`` method.
4. an instance of an :class:`.F2FSynthesisPlugin`, which is used as is. This is how a plugin that is
   not registered through an entry point enters the pass.
"""


class F2FSynthesis(FermionicDAGCircuitPass):
    r"""A transpilation pass breaking gates down into smaller ones in fermionic space.

    Some gates can be rewritten into a circuit of smaller fermionic gates before they reach the
    fermion-to-qubit stage, which then maps and synthesizes those. This pass does that for the gates
    named in :attr:`methods`, with the plugin that :attr:`methods` names for each, so the choice is
    made once for a whole transpilation pipeline instead of gate by gate. It is the fermionic
    counterpart of :class:`.F2QSynthesis`, one stage earlier: where that pass maps every instruction
    to qubits, this one rewrites some gates into others and leaves the rest as they are.

    .. code-block:: python

       pm.optimization = FermionicPassManager(
           [
               MergeOrbitalRotations(),
               MergeSlaterDeterminantPreparation(),
               F2FSynthesis({"OrbitalRotation": "Givens", "PrepareSlaterDeterminant": "Givens"}),
           ]
       )

    .. rubric:: Usage

    There are two ways to configure the plugins used by this transpiler pass:

    1. Providing a ``config`` during initialization:

       .. doctest::

          >>> from qiskit_fermions.transpiler import passes
          >>>
          >>> config = {"Evolution": ("SuzukiTrotter", (), {"order": 2})}
          >>> synth = passes.F2FSynthesis(config)

    2. Manually populating the :attr:`methods` attribute:

       .. doctest::

          >>> from qiskit_fermions.circuit.library import synthesis
          >>> from qiskit_fermions.transpiler import passes
          >>>
          >>> synth = passes.F2FSynthesis()
          >>> synth.methods["Evolution"] = synthesis.FermionicSuzukiTrotter(order=2)

       Through this manual approach it is also possible to inject custom
       :class:`.F2FSynthesisPlugin` implementations without requiring them to be registered through
       an `entry-point <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_.

    It belongs in the ``optimization`` stage, ahead of the layout and the synthesis, and `after` the
    passes that consume these gates as such. Placing it earlier would break the gates up before those
    passes can fuse them, which silently gives up the cheaper Slater-determinant decomposition (see
    :class:`.MergeSlaterDeterminantPreparation`).

    Each selected gate is replaced by the circuit that its plugin produces. A gate whose type is not a
    key of :attr:`methods` is left untouched, as is any gate that is :attr:`~.FermionicGate.atomic`
    (also with ``apply=False``: it is not even tagged with the plugin)
    and any node rejected by an optional :attr:`filter`. A gate that has no synthesis at all, such as
    an :class:`.InitializeModes`, passes through no matter what.

    .. important::
       What the replacement is equivalent to is specified by the gate and its plugin, not by this
       pass: usually it implements the same unitary (or approximates it), but a gate may specify
       less, and then only a circuit that meets that weaker specification may use this pass. Consult
       the documentation of the gates you configure, and see :class:`.F2FSynthesisPlugin`.

    .. note::
       Running the pass again is harmless: the factors a plugin emits are
       :attr:`~.FermionicGate.atomic`, and an atomic gate is a terminal factor which this pass leaves
       in place rather than splitting further. The same exemption makes the pass a fixed point over
       the output of :class:`.QDriftTrotterization`, whose sampled gates are atomic because the random
       draw *is* the Trotterization it performs.

    .. note::
       Set :attr:`apply` to ``False`` to only *select* the method and leave the expansion to
       something else, such as a :meth:`~.FermionicCircuit.decompose` call. Be aware that a gate which
       is never expanded reaches the fermion-to-qubit stage whole, where it is mapped without its
       :attr:`~.FermionicGate.synthesis` ever being read, and every method then produces identical
       output.

    .. caution::
       Not every method suits every operator. An :class:`.Evolution` whose operator groups all
       mutually commute (the diagonal-Coulomb operators of a :class:`.UCJ`, for example) is
       synthesized exactly at any order, so a higher order only adds depth. Use :attr:`filter` to
       restrict the pass to the gates that benefit.

       Expanding an operator that carries no :attr:`~qiskit_fermions.operators.OperatorTrait.groups`
       splits it term by term, and a lone hopping term :math:`a^\dagger_i a_j` is not Hermitian, so
       its exponential is not unitary. Nothing complains until the mapped operator reaches
       :class:`~qiskit.circuit.library.PauliEvolutionGate`, which raises ``ValueError: Operator
       contains complex coefficients, which are not supported``. Assigning conjugate-paired groups
       is what makes each factor Hermitian; see
       :func:`~qiskit_fermions.operators.terms.grouping.groups_are_hermitian`.
    """

    def __init__(
        self,
        config: F2FSynthesisConfig | None = None,
        *,
        apply: bool = True,
        filter: Callable[[DAGOpNode], bool] | None = None,  # noqa: A002
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            config: an optional dictionary to pre-populate the :attr:`methods` with plugins provided
                by the :class:`.F2FSynthesisPluginManager`. See :type:`.F2FSynthesisConfig`.
            apply: whether to expand each selected gate into the factors its plugin produces. When
                ``False``, the gate is only tagged with the plugin (unless it is atomic, which is
                never touched) and something else has to expand it later. See :attr:`apply`.
            filter: an optional predicate deciding which nodes to break down. It is called with the
                :class:`~qiskit.dagcircuit.DAGOpNode` of a gate whose type is in :attr:`methods`,
                and the node is left untouched unless it returns ``True``. If ``None`` (the default),
                every such node is selected.

        Raises:
            KeyError: if ``config`` names a gate or a method for which no plugin is installed.
            TypeError: if ``config`` gives a value that is none of the forms of
                :type:`.F2FSynthesisConfig`.
        """
        super().__init__()

        self.methods: dict[str, F2FSynthesisPlugin] = {}
        """A dictionary of the fermion-to-fermion synthesis methods, by gate.

        The keys of this dictionary correspond to the ``__name__`` of a :class:`.FermionicGate`, and
        the values are instances of the :class:`.F2FSynthesisPlugin` protocol. A gate that is not a
        key is not touched by this pass, so the dictionary is **empty** by default, which makes the
        pass do nothing. Pre-populate it with plugins provided by the :class:`.F2FSynthesisPluginManager`
        through the ``config`` argument.
        """

        if config is not None:
            manager = F2FSynthesisPluginManager()
            for op_name, method_spec in config.items():
                init_args: Sequence[Any] = ()
                init_kwargs: Mapping[str, Any] = {}
                match method_spec:
                    case str():
                        method_name = method_spec
                    case (str() as name, args):
                        method_name = name
                        init_args = args
                    case (str() as name, args, kwargs):
                        method_name = name
                        init_args = args
                        init_kwargs = kwargs
                    case _ if callable(getattr(method_spec, "synthesize", None)):
                        self.methods[op_name] = method_spec
                        continue
                    case _:
                        raise TypeError(
                            f"The synthesis for '{op_name}' must be a method name, a tuple of a "
                            f"method name and its arguments, or a plugin instance, not {method_spec!r}."
                        )

                if op_name not in manager.plugins:
                    raise KeyError(
                        f"No fermion-to-fermion synthesis is installed for '{op_name}'. Gates that "
                        f"have one: {sorted(manager.op_names())}."
                    )
                if method_name not in manager.method_names(op_name):
                    raise KeyError(
                        f"No synthesis method '{method_name}' is installed for '{op_name}'. "
                        f"Installed: {sorted(manager.method_names(op_name))}."
                    )

                plugin = manager.method(op_name, method_name)
                self.methods[op_name] = plugin(*init_args, **init_kwargs)

        self.apply = apply
        """Whether to expand each selected gate into the factors its plugin produces.

        When this is ``False``, the pass only *selects* the method: it tags each selected gate with
        its plugin (see :meth:`.FermionicGate.with_synthesis`), except an :attr:`~.FermionicGate.atomic`
        gate, which is left as it is, and leaves the expansion to whatever
        builds the gate's definition later, such as a :meth:`~.FermionicCircuit.decompose` call. Note
        that a gate which is never expanded reaches the fermion-to-qubit stage whole, where it is
        mapped without its :attr:`~.FermionicGate.synthesis` ever being read.
        """

        self.filter = filter
        """The predicate selecting which nodes to break down."""

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Every node whose gate type is in :attr:`methods` and that is accepted by :attr:`filter` is
        replaced by the gates that its plugin produces, or (when :attr:`apply` is ``False``) by an
        equivalent gate merely carrying the plugin. All other nodes are left untouched. The input DAG
        is modified in place.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported, plus
                :class:`~qiskit.circuit.library.Barrier` and :class:`.FermionicMeasure`, which are
                carried through untouched (a measurement keeps the classical bit it writes into).

        Returns:
            The output circuit which is still acting on a fermionic register.
        """
        # Circuit order, not storage order: the latter is out of sync with the wire dependencies
        # after a pass substituted a node by a sub-circuit, and this pass does so itself.
        # Materialized with `list` because the loop substitutes nodes in place. Qiskit documents
        # `topological_op_nodes` as a generator, which must not be consumed while the DAG changes
        # under it; the current implementation happens to hand out an already collected sequence,
        # but that is not a guarantee. Substituting one node leaves the others valid.
        for node in list(dag.topological_op_nodes()):
            if not isinstance(node.op, FermionicGate):
                continue
            # Key on `base_class`, as `F2QSynthesis` does: it is the user-facing class, identical to
            # `type(...)` for every non-singleton gate and robust against generated subclasses.
            plugin = self.methods.get(node.op.base_class.__name__)
            if plugin is None:
                continue

            # An atomic gate is a terminal factor that must not be broken down any further, so it
            # passes through unchanged: neither expanded nor tagged with the plugin, and not even
            # offered to the filter. This is also what keeps the pass a fixed point over its own
            # output and over `QDriftTrotterization`'s.
            if node.op.atomic:
                continue

            if self.filter is not None and not self.filter(node):
                continue

            if not self.apply:
                # A *copy* rather than an assignment to `node.op`: a gate instance can be shared
                # between circuits (Qiskit copies gates with a shallow `__dict__` copy), so mutating
                # it would retroactively change the synthesis of every other circuit holding it.
                dag.substitute_node(node, node.op.with_synthesis(plugin), inplace=True)
                continue

            # Synthesize directly rather than going through the gate's definition: the factors are
            # all that is wanted here, and for a gate whose synthesis is not its definition (the
            # Slater determinant) the definition is not what is wanted at all.
            expanded = circuit_to_dag(plugin.synthesize(node.op)._inner)
            dag.substitute_node_with_dag(node, expanded)

        return dag
