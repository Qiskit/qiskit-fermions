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

"""Qiskit Fermions.

..
   Refer to ``docs/pydoc/index.rst`` for the actual documentation of this module.
"""

import sys
from inspect import ismodule

from . import _lib  # type: ignore[attr-defined]

__modules = {_lib: "qiskit_fermions._lib"}

while len(__modules):
    for __module, __path in __modules.copy().items():
        for __submodule_name in __module.__all__:
            __submodule = getattr(__module, __submodule_name)
            __submodule_path = f"{__path}.{__submodule_name}"
            if ismodule(__submodule):
                __modules[__submodule] = __submodule_path
                sys.modules[__submodule_path] = __submodule
        del __modules[__module]
