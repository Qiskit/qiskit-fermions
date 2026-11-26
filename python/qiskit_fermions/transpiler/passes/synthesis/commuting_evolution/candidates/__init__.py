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

"""The built-in candidate syntheses."""

from __future__ import annotations

from .base import Candidate
from .clifford import Majority, Plain, TwinFold
from .local import Local
from .search import Search

DEFAULT_CANDIDATES: tuple[Candidate, ...] = (
    Plain(),
    Majority(),
    TwinFold(),
    Local(),
    Search(),
)
"""The candidates the plugin uses by default, in the order in which ties are broken: every built-in
candidate.

:class:`.Search`, the most expensive one, comes last, so that it decides only where it is strictly
better, and it runs only for a gate that is not placed on the device.
"""

__all__ = [
    "DEFAULT_CANDIDATES",
    "Candidate",
    "Local",
    "Majority",
    "Plain",
    "Search",
    "TwinFold",
]
