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

"""Fermion-to-fermion gate synthesis plugin management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from stevedore import ExtensionManager

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit, FermionicGate


class F2FSynthesisPlugin(Protocol):
    r"""The protocol for plugins to the :class:`.F2FSynthesis` transpiler pass.

    A plugin rewrites a gate into a :class:`.FermionicCircuit` of *smaller* :class:`.FermionicGate`
    instances. Both sides of that rewrite live in `fermionic` space, so no encoding is involved: how
    the smaller gates are then mapped to qubits is left to the fermion-to-qubit stage (see
    :class:`.F2QSynthesis`). It is the counterpart of :class:`.F2QSynthesisPlugin` one step earlier.

    The gates a plugin emits should be marked :attr:`~.FermionicGate.atomic`, and an
    :class:`.Evolution` among them should carry an operator with a single group, so that the
    fermion-to-qubit stage maps every one of them whole.

    .. important::
       What the circuit has to be equivalent to is specified by the *gate* a plugin is registered
       for, not by this protocol, and it may differ between gates: a circuit that implements the same
       unitary is the common case, but a gate may settle for less (for example, the same state when
       prepared from the vacuum). Consult the documentation of the gate whose method you implement,
       and document any such caveat there.

    .. important::
       This fermion-to-fermion step is **optional**. A gate can be handed to the fermion-to-qubit
       stage whole, where a plugin of that stage registered for it can map it directly.
    """

    def synthesize(self, gate: FermionicGate) -> FermionicCircuit:
        """Synthesizes the provided gate.

        Args:
            gate: the gate to synthesize, which is of the one type that this plugin is registered for
                (the first part of its entry-point name).

        Returns:
            A :class:`.FermionicCircuit` on ``gate.num_modes`` modes implementing the gate, in the
            sense that the documentation of the gate defines.
        """
        ...


class F2FSynthesisPluginManager:
    """A simple manager of the installed fermion-to-fermion synthesis plugins.

    All plugins that are registered under the ``qiskit_fermions.circuit.synthesis``
    `entry-point <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_ will be
    available to this class and get exposed by the :attr:`plugins` attribute. The name of an entry
    point is the name of the gate class followed by the name of the method, separated by a dot, for
    example ``OrbitalRotation.Givens``. Below is the table from the ``pyproject.toml`` of this package
    itself:

    .. literalinclude:: ../../pyproject.toml
       :language: toml
       :start-at: qiskit_fermions.circuit.synthesis
       :end-before: [

    An overview of all builtin plugins is given :ref:`here <builtin_f2f_synthesis_plugins>`.
    """

    def __init__(self) -> None:  # noqa: D107
        self._manager = ExtensionManager(
            "qiskit_fermions.circuit.synthesis",
            invoke_on_load=False,
            propagate_map_exceptions=True,
        )

        self.plugins: dict[str, list[str]] = {}
        """A dictionary mapping gate names to their synthesis methods.

        The keys of this dictionary are names of :class:`.FermionicGate` classes. The values are the
        list of installed :class:`.F2FSynthesisPlugin` methods that may be used to synthesize the
        instances of that particular gate.
        """

        for plugin_name in self._manager.names():
            op_name, method_name = plugin_name.split(".")
            if op_name not in self.plugins:
                self.plugins[op_name] = []
            self.plugins[op_name].append(method_name)

    def method_names(self, op_name: str) -> list[str]:
        """Returns the names of all installed plugin methods for a given gate.

        Args:
            op_name: the name of the gate whose synthesis methods to return.

        Returns:
            The list of installed plugin method names.
        """
        return self.plugins.get(op_name, [])

    def method(self, op_name: str, method_name: str) -> type[F2FSynthesisPlugin]:
        """Returns the requested :class:`.F2FSynthesisPlugin` type.

        Args:
            op_name: the name of the gate whose plugin to return.
            method_name: the name of the synthesis plugin.

        Returns:
            The :class:`.F2FSynthesisPlugin` class.

        Raises:
            KeyError: if no plugin is registered under that name.
        """
        plugin_name = op_name + "." + method_name
        return cast(type[F2FSynthesisPlugin], self._manager[plugin_name].plugin)

    def op_names(self) -> list[str]:
        """Returns the names of all gates for which any plugin is installed."""
        return list(self.plugins.keys())
