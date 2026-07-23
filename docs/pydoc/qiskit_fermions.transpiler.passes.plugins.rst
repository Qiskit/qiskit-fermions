.. _qiskit_fermions-transpiler-passes-synthesis-plugins:

=======================
Transpiler Pass Plugins
=======================

.. currentmodule:: qiskit_fermions.transpiler.passes.synthesis

The :class:`.F2QSynthesis` transpiler pass exposes a `plugin interface` (see
:class:`.F2QSynthesisPlugin`) through which implementations for mapping fermion-based instructions
to qubit-based ones can be registered.

The available plugins are managed by the :class:`.F2QSynthesisPluginManager`. Choosing the right
combination of synthesis plugins is crucial and the :class:`.F2QSynthesis` transpilation pass does
**not** have any default plugins defined! For this, refer to the transpiler
:mod:`~qiskit_fermions.transpiler.presets`.

.. autosummary::
   :toctree: ../stubs/

   F2QSynthesisPlugin
   F2QSynthesisPluginManager

The plugins to use at runtime have to be registered in the :attr:`.F2QSynthesis.methods` attribute.
This can be done easily through the ``config`` argument of :class:`.F2QSynthesis` which specifies
the plugins to use from the :class:`.F2QSynthesisPluginManager` or it can be done manually by
writing into :attr:`.F2QSynthesis.methods` directly.

The table below lists the builtin synthesis plugins, organized by the :class:`.FermionicGate` that
they act upon:

.. _builtin_synthesis_plugins:

.. table:: An overview of the builtin ``qiskit_fermions.transpiler.synthesis`` entry-points.

   +------------------------------------+-------------------------------+--------------------------------------------------------+
   | Circuit instruction name           | Plugin method name            | Reference                                              |
   +====================================+===============================+========================================================+
   | :class:`.Evolution`                | ``MapperFn``                  | :class:`MapperFnEvolutionSynthesis`                    |
   +------------------------------------+-------------------------------+--------------------------------------------------------+
   | :class:`.InitializeModes`          | ``TrivialOccupation``         | :class:`TrivialOccupationInitializeModesSynthesis`     |
   +------------------------------------+-------------------------------+--------------------------------------------------------+
   | :class:`.OrbitalRotation`          | ``GivensDecomposition``       | :class:`GivensDecompositionOrbitalRotationSynthesis`   |
   +------------------------------------+-------------------------------+--------------------------------------------------------+
   | :class:`.PrepareSlaterDeterminant` | ``GivensDecompositionSlater`` | :class:`GivensDecompositionSlaterDeterminantSynthesis` |
   +------------------------------------+-------------------------------+--------------------------------------------------------+

.. autosummary::
   :toctree: ../stubs/
   :class: sd-d-none

   GivensDecompositionOrbitalRotationSynthesis
   GivensDecompositionSlaterDeterminantSynthesis
   MapperFnEvolutionSynthesis
   TrivialOccupationInitializeModesSynthesis
