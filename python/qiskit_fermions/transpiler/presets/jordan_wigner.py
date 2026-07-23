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

from qiskit_fermions.mappers.library import jordan_wigner

from .. import FermionicCircuitToDAG, FermionicPassManager, QuantumDAGToCircuit
from ..passes import (
    F2QSynthesis,
    F2QSynthesisConfig,
    MergeOrbitalRotations,
    MergeSlaterDeterminantPreparation,
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
    # First merge any run of consecutive OrbitalRotation gates into a single rotation, so the
    # synthesis stage below lowers one decomposition per run rather than one per gate. Then fuse an
    # InitializeModes followed by an OrbitalRotation into a single PrepareSlaterDeterminant so that
    # synthesis can lower it with the reduced Slater decomposition rather than the full square
    # orbital rotation. The order matters: merging the rotations first exposes the single rotation
    # immediately following the initialization, which the Slater-prep fusion then contracts into it.
    optimization = FermionicPassManager(
        [MergeOrbitalRotations(), MergeSlaterDeterminantPreparation()]
    )

    layout = FermionicPassManager(TrivialF2QLayout())

    config: F2QSynthesisConfig = {
        "Evolution": ("MapperFn", (jordan_wigner,)),
        "InitializeModes": "TrivialOccupation",
        "OrbitalRotation": "GivensDecomposition",
        "PrepareSlaterDeterminant": "GivensDecompositionSlater",
    }
    synth = F2QSynthesis(config)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=optimization,
        layout=layout,
        synthesis=synth,
        qubit=generate_preset_pass_manager(**kwargs),
        output=QuantumDAGToCircuit(),
    )

    return pm
