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

"""Configuration of Sybil doctests integration for pytest."""

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.evaluators.doctest import NUMBER
from sybil.parsers.rest import DocTestParser, SkipParser

pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE | NUMBER),
        # ``SkipParser`` lets a docstring gate its doctest on an optional dependency, the same way
        # the guides under ``docs/`` do. Needed because ffsim has no Windows support.
        SkipParser(),
    ],
    patterns=["*.py", "*.pyi"],
).pytest()
