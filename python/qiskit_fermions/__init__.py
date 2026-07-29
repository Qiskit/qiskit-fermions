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
from inspect import isclass, ismodule

from . import _lib
from .version import __version__  # noqa: F401

__modules = {_lib: "qiskit_fermions._lib"}

while len(__modules):
    for __module, __path in __modules.copy().items():
        for __submodule_name in __module.__all__:
            __submodule = getattr(__module, __submodule_name)
            __submodule_path = f"{__path}.{__submodule_name}"
            if ismodule(__submodule):
                __modules[__submodule] = __submodule_path
                sys.modules[__submodule_path] = __submodule
            elif isclass(__submodule) and __submodule.__module__ != __path:
                # Native pyclasses declare their public, logical dotted path (e.g.
                # ``qiskit_fermions.operators.fermion_operator``) as ``__module__`` for a
                # user-facing ``repr``/error messages, which does not match the physical
                # ``_lib``-rooted path under which they are actually importable. That
                # logical path is never itself a real module, so ``pickle`` cannot resolve
                # it via ``importlib.import_module`` when serializing a reference to the
                # class. Aliasing it to the real (private, underscore-prefixed) module here
                # fixes that without touching the public-facing ``__module__``.
                sys.modules[__submodule.__module__] = __module
        del __modules[__module]
