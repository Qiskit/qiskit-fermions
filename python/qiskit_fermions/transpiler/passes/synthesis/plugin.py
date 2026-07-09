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

"""Fermion-to-qubit circuit synthesis pass plugin management."""

from __future__ import annotations

from typing import Protocol, cast

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from stevedore import ExtensionManager

from ... import F2QLayout


class F2QSynthesisPlugin(Protocol):
    """The protocol for plugins to the :class:`.F2QSynthesis` transpiler pass."""

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        """Translates the provided fermion-based circuit instruction to a qubit-based one.

        Args:
            in_node: a fermion-based circuit instruction stored in a
                :class:`~qiskit.dagcircuit.DAGOpNode`. Specifically, this guarantees that
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` is of type :class:`.FermionicGate`.
            out_dag: the qubit-based :class:`~qiskit.dagcircuit.DAGCircuit` into which this plugin
                must insert the translated circuit instruction.
            f2q_layout: the :type:`~qiskit_fermions.transpiler.F2QLayout` setting that is global to
                the transpilation process. It is the plugin's responsibility to respect this mapping
                of :type:`~qiskit_fermions.circuit.FermionicRegister` to
                :class:`~qiskit.circuit.QuantumRegister`.
        """
        ...


class F2QSynthesisPluginManager:
    """A simple manager of the installed fermion-to-qubit synthesis plugins.

    All plugins that are registered under the ``qiskit_fermions.transpiler.synthesis``
    `entry-point <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_ will be
    available to this class and get exposed by the :attr:`plugins` attribute. Below is an example
    from the ``pyproject.toml`` of this package itself:

    .. literalinclude:: ../../pyproject.toml
       :language: toml
       :start-at: entry-points
       :end-before: setuptools

    An overview of all builtin plugins is given in :ref:`this table <builtin_synthesis_plugins>`.
    """

    def __init__(self) -> None:  # noqa: D107
        self._manager = ExtensionManager(
            "qiskit_fermions.transpiler.synthesis",
            invoke_on_load=False,
            propagate_map_exceptions=True,
        )

        self.plugins: dict[str, list[str]] = {}
        """A dictionary mapping circuit instruction names to their synthesis methods.

        The keys of this dictionary are names of :class:`.FermionicGate` classes. The values are the
        list of installed :class:`.F2QSynthesisPlugin` methods that may be used to synthesize the
        instances of that particular gate to its qubit form.
        """

        for plugin_name in self._manager.names():
            op_name, method_name = plugin_name.split(".")
            if op_name not in self.plugins:
                self.plugins[op_name] = []
            self.plugins[op_name].append(method_name)

    def method_names(self, op_name: str) -> list[str]:
        """Returns the names of all installed plugin methods for a given circuit instruction.

        Args:
            op_name: the name of the operation whose synthesis methods to return.

        Returns:
            The list of installed plugin method names.
        """
        return self.plugins.get(op_name, [])

    def method(self, op_name: str, method_name: str) -> type[F2QSynthesisPlugin]:
        """Returns the requested :class:`.F2QSynthesisPlugin` type.

        Args:
            op_name: the name of the operation whose plugin to return.
            method_name: the name of the synthesis plugin.

        Returns:
            The :class:`.F2QSynthesisPlugin` class.
        """
        plugin_name = op_name + "." + method_name
        return cast(type[F2QSynthesisPlugin], self._manager[plugin_name].plugin)

    def op_names(self) -> list[str]:
        """Returns the names of all circuit instructions for which any plugin is installed."""
        return list(self.plugins.keys())
