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

"""The preset transpiler pipeline based on the Jordan-Wigner fermion-to-qubit mapping."""

from qiskit.passmanager import MultiStagePassManager
from qiskit.transpiler import generate_preset_pass_manager

from qiskit_fermions.circuit.library import Evolution, InitializeModes, OrbitalRotation
from qiskit_fermions.mappers.library import jordan_wigner

from .. import FermionicCircuitToDAG, FermionicPassManager, QuantumDAGToCircuit
from ..passes import (
    EvolutionSynthesis,
    F2QSynthesis,
    InitializeModesSynthesis,
    OrbitalRotationSynthesis,
    TrivialF2QLayout,
)


def generate_preset_jw_pass_manager(**kwargs) -> MultiStagePassManager:
    """Generates a preset transpiler pipeline based on the :func:`.jordan_wigner` mapping.

    Args:
        kwargs: any additional keyword arguments are forwarded to
            :external:func:`~qiskit.transpiler.generate_preset_pass_manager` whose output is used
            for the ``qubit`` stage.

    Returns:
        The preset staged fermion-to-qubit transpiler pipeline.
    """
    optimization = FermionicPassManager()

    layout = FermionicPassManager(TrivialF2QLayout())

    synth = F2QSynthesis()
    synth.plugins[Evolution] = EvolutionSynthesis(jordan_wigner)  # type: ignore[arg-type]
    synth.plugins[InitializeModes] = InitializeModesSynthesis()
    synth.plugins[OrbitalRotation] = OrbitalRotationSynthesis()

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=optimization,
        layout=layout,
        synthesis=synth,
        qubit=generate_preset_pass_manager(**kwargs),
        output=QuantumDAGToCircuit(),
    )

    return pm
