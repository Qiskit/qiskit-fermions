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

import copy

from qiskit.passmanager import MultiStagePassManager
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

from qiskit_fermions.mappers.library import jordan_wigner

from .. import FermionicCircuitToDAG, FermionicPassManager, QuantumDAGToCircuit
from ..passes import (
    F2FSynthesis,
    F2QSynthesis,
    F2QSynthesisConfig,
    HopEvolutionSynthesis,
    MergeOrbitalRotations,
    MergeSlaterDeterminantPreparation,
    TrivialF2QLayout,
)


def _with_hop_synthesis(hls_config: HLSConfig | None) -> HLSConfig:
    """Returns a copy of ``hls_config`` with the hop synthesis tried first for Pauli evolutions.

    The methods that ``hls_config`` lists for ``PauliEvolution`` keep their order behind the hop
    synthesis. If it lists none, the default method follows, as it would without a configuration.
    """
    if hls_config is None:
        return HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"])

    merged = copy.copy(hls_config)
    merged.methods = dict(hls_config.methods)

    existing = merged.methods.get("PauliEvolution")
    if existing is None:
        existing = ["default"] if hls_config.use_default_on_unspecified else []
    elif not isinstance(existing, list):
        existing = [existing]

    merged.methods["PauliEvolution"] = [HopEvolutionSynthesis(), *existing]
    return merged


def generate_preset_jw_pass_manager(**kwargs) -> MultiStagePassManager:
    """Generates a preset transpiler pipeline based on the :func:`.jordan_wigner` mapping.

    Args:
        kwargs: any additional keyword arguments are forwarded to
            :external:func:`~qiskit.transpiler.generate_preset_pass_manager` whose output is used
            for the ``qubit`` stage. The ``hls_config`` is extended, not replaced, so that a
            :class:`~qiskit.circuit.library.PauliEvolutionGate` of a hop between two qubits is
            **always** synthesized by the :class:`.HopEvolutionSynthesis`, ahead of whatever methods
            the given configuration lists for it (``"default"`` included, which then only applies to
            what is not a hop). The given configuration is not modified. A pipeline that should not
            do this has to be assembled from its stages rather than taken from this preset.

    Returns:
        The preset staged fermion-to-qubit transpiler pipeline.
    """
    kwargs["hls_config"] = _with_hop_synthesis(kwargs.get("hls_config"))

    # First merge any run of consecutive OrbitalRotation gates into a single rotation, so the
    # synthesis stage below lowers one decomposition per run rather than one per gate. Then fuse an
    # InitializeModes followed by an OrbitalRotation into a single PrepareSlaterDeterminant so that
    # synthesis can lower it with the reduced Slater decomposition rather than the full square
    # orbital rotation. The order matters: merging the rotations first exposes the single rotation
    # immediately following the initialization, which the Slater-prep fusion then contracts into it.
    #
    # Whatever orbital rotation is left after that is decomposed in fermionic space into Givens
    # factors. This must come last: decomposing earlier would break the rotations up before the two
    # merge passes above can fuse them, silently giving up the cheaper Slater decomposition. Only the
    # gates named here are touched; in particular an `Evolution` is mapped to qubits whole.
    optimization = FermionicPassManager(
        [
            MergeOrbitalRotations(),
            MergeSlaterDeterminantPreparation(),
            F2FSynthesis({"OrbitalRotation": "Givens"}),
        ]
    )

    layout = FermionicPassManager(TrivialF2QLayout())

    config: F2QSynthesisConfig = {
        "Evolution": ("MapperFn", (jordan_wigner,)),
        "FermionicMeasure": "TrivialOccupation",
        "InitializeModes": "TrivialOccupation",
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
