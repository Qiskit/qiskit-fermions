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

"""The first-order Lie-Trotter fermionic product formula."""

from __future__ import annotations

from .suzuki_trotter import FermionicSuzukiTrotter


class FermionicLieTrotter(FermionicSuzukiTrotter):
    r"""The first-order Lie-Trotter product formula, applied in fermionic space.

    Given an :class:`.Evolution` gate with the evolution time :math:`t`, this synthesizes the
    approximation

    .. math::

        e^{-i t \sum_j H_j} \approx \prod_j e^{-i t H_j},

    emitting one :class:`.Evolution` gate per factor :math:`H_j`, each evolving for the whole time
    :math:`t` -- or, with :attr:`~.FermionicLieTrotter.reps` repetitions, for :math:`t/\texttt{reps}`
    apiece across that many sweeps.

    The operator is split into the :math:`H_j`, and each factor narrowed to the modes it actually
    touches, exactly as described for :class:`.FermionicSuzukiTrotter`.

    First order is the degenerate case of the Suzuki recursion, so this is implemented as a
    :class:`.FermionicSuzukiTrotter` with the order fixed to ``1``; the two are interchangeable at
    equal :attr:`~.FermionicLieTrotter.reps`. It keeps a name of its own because it is the default
    synthesis of an :class:`.Evolution` gate, and because a plain first-order product formula is more
    recognizable under that name than as an argument to a higher-order one.

    .. note::
       This product formula is `exact` only when the factors mutually commute. Grouping an operator so
       that each group collects mutually commuting terms therefore both shortens the circuit and
       reduces the Trotter error. Where the factors do not commute, an even
       :attr:`~.FermionicSuzukiTrotter.order` reduces the error further at the same grouping.

       Note also that applying it at all is optional: leaving the gate undecomposed and letting the
       fermion-to-qubit stage handle the whole operator incurs no Trotter error at this level. See
       :mod:`~qiskit_fermions.circuit.library.synthesis`.

    The Hermiticity requirement on the individual factors, described in the
    :class:`.FermionicSuzukiTrotter` documentation, applies here too.
    """

    def __init__(self, reps: int = 1) -> None:
        """Initializing an instance of this synthesis method can be done with the argument below.

        Args:
            reps: the number of times to repeat the formula, each repetition evolving for
                ``time / reps``.

        Raises:
            ValueError: if ``reps`` is not positive.
        """
        super().__init__(order=1, reps=reps)
