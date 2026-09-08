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

from pathlib import Path

import pytest
from qiskit_fermions.operators import FermionOperator, ann, cre
from qiskit_fermions.operators.library import FCIDump


class TestFCIDump:
    def test_from_file(self):
        file_path = Path(__file__).parent / "../../../h2.fcidump"
        fcidump = FCIDump.from_file(str(file_path))
        assert fcidump.norb == 2
        assert fcidump.nelec == 2
        assert fcidump.ms2 == 0
        op = FermionOperator.from_fcidump(fcidump)
        expected = FermionOperator.from_dict(
            {
                (): 0.71996899444897966,
                (cre(0), ann(0)): -1.2563390730032502,
                (cre(1), ann(1)): -0.4718960072811406,
                (cre(2), ann(2)): -1.2563390730032502,
                (cre(3), ann(3)): -0.4718960072811406,
                (cre(0), cre(0), ann(0), ann(0)): 0.33785507740175824,
                (cre(0), cre(1), ann(1), ann(0)): 0.3322908651276483,
                (cre(0), cre(2), ann(2), ann(0)): 0.33785507740175824,
                (cre(0), cre(3), ann(3), ann(0)): 0.3322908651276483,
                (cre(0), cre(0), ann(1), ann(1)): 0.09046559989211567,
                (cre(0), cre(1), ann(0), ann(1)): 0.09046559989211567,
                (cre(0), cre(2), ann(3), ann(1)): 0.09046559989211567,
                (cre(0), cre(3), ann(2), ann(1)): 0.09046559989211567,
                (cre(1), cre(0), ann(1), ann(0)): 0.09046559989211567,
                (cre(1), cre(1), ann(0), ann(0)): 0.09046559989211567,
                (cre(1), cre(2), ann(3), ann(0)): 0.09046559989211567,
                (cre(1), cre(3), ann(2), ann(0)): 0.09046559989211567,
                (cre(1), cre(0), ann(0), ann(1)): 0.3322908651276483,
                (cre(1), cre(1), ann(1), ann(1)): 0.34928686136600917,
                (cre(1), cre(2), ann(2), ann(1)): 0.3322908651276483,
                (cre(1), cre(3), ann(3), ann(1)): 0.34928686136600917,
                (cre(2), cre(0), ann(0), ann(2)): 0.33785507740175824,
                (cre(2), cre(1), ann(1), ann(2)): 0.3322908651276483,
                (cre(2), cre(2), ann(2), ann(2)): 0.33785507740175824,
                (cre(2), cre(3), ann(3), ann(2)): 0.3322908651276483,
                (cre(2), cre(0), ann(1), ann(3)): 0.09046559989211567,
                (cre(2), cre(1), ann(0), ann(3)): 0.09046559989211567,
                (cre(2), cre(2), ann(3), ann(3)): 0.09046559989211567,
                (cre(2), cre(3), ann(2), ann(3)): 0.09046559989211567,
                (cre(3), cre(0), ann(1), ann(2)): 0.09046559989211567,
                (cre(3), cre(1), ann(0), ann(2)): 0.09046559989211567,
                (cre(3), cre(2), ann(3), ann(2)): 0.09046559989211567,
                (cre(3), cre(3), ann(2), ann(2)): 0.09046559989211567,
                (cre(3), cre(0), ann(0), ann(3)): 0.3322908651276483,
                (cre(3), cre(1), ann(1), ann(3)): 0.34928686136600917,
                (cre(3), cre(2), ann(2), ann(3)): 0.3322908651276483,
                (cre(3), cre(3), ann(3), ann(3)): 0.34928686136600917,
            }
        )
        assert op.equiv(expected)

    def test_from_file_beta(self):
        file_path = Path(__file__).parent / "../../../heh.fcidump"
        fcidump = FCIDump.from_file(str(file_path))
        assert fcidump.norb == 2
        assert fcidump.nelec == 3
        assert fcidump.ms2 == 1
        op = FermionOperator.from_fcidump(fcidump)
        expected = FermionOperator.from_dict(
            {
                (): 1.4399379888979593,
                (cre(0), ann(0)): -2.6053045895340987,
                (cre(0), ann(1)): 0.18301050723224974,
                (cre(1), ann(0)): 0.18301050723224993,
                (cre(1), ann(1)): -1.3466434111981145,
                (cre(2), ann(2)): -2.6172710340816154,
                (cre(2), ann(3)): 0.13523295000711089,
                (cre(3), ann(2)): 0.13523295000711105,
                (cre(3), ann(3)): -1.334676966650596,
                (cre(0), cre(0), ann(0), ann(0)): 0.46921909323587185,
                (cre(0), cre(0), ann(0), ann(1)): -0.08590822896973446,
                (cre(0), cre(0), ann(1), ann(0)): -0.08590822896973446,
                (cre(0), cre(0), ann(1), ann(1)): 0.07642066474103536,
                (cre(0), cre(1), ann(0), ann(0)): -0.08590822896973446,
                (cre(0), cre(1), ann(0), ann(1)): 0.07642066474103536,
                (cre(0), cre(1), ann(1), ann(0)): 0.3394457138157064,
                (cre(0), cre(1), ann(1), ann(1)): 0.014617877034409275,
                (cre(0), cre(2), ann(2), ann(0)): 0.4754873109620925,
                (cre(0), cre(2), ann(2), ann(1)): -0.0915052536161256,
                (cre(0), cre(2), ann(3), ann(0)): -0.08079285521616332,
                (cre(0), cre(2), ann(3), ann(1)): 0.07243027144542874,
                (cre(0), cre(3), ann(2), ann(0)): -0.08079285521616332,
                (cre(0), cre(3), ann(2), ann(1)): 0.07243027144542874,
                (cre(0), cre(3), ann(3), ann(0)): 0.3331774960894852,
                (cre(0), cre(3), ann(3), ann(1)): 0.020214901680800382,
                (cre(1), cre(0), ann(0), ann(0)): -0.08590822896973446,
                (cre(1), cre(0), ann(0), ann(1)): 0.3394457138157064,
                (cre(1), cre(0), ann(1), ann(0)): 0.07642066474103536,
                (cre(1), cre(0), ann(1), ann(1)): 0.014617877034409275,
                (cre(1), cre(1), ann(0), ann(0)): 0.07642066474103536,
                (cre(1), cre(1), ann(0), ann(1)): 0.014617877034409275,
                (cre(1), cre(1), ann(1), ann(0)): 0.014617877034409275,
                (cre(1), cre(1), ann(1), ann(1)): 0.3767367167065582,
                (cre(1), cre(2), ann(2), ann(0)): -0.0915052536161256,
                (cre(1), cre(2), ann(2), ann(1)): 0.33840060689985296,
                (cre(1), cre(2), ann(3), ann(0)): 0.07243027144542874,
                (cre(1), cre(2), ann(3), ann(1)): 0.013176380212608446,
                (cre(1), cre(3), ann(2), ann(0)): 0.07243027144542874,
                (cre(1), cre(3), ann(2), ann(1)): 0.013176380212608446,
                (cre(1), cre(3), ann(3), ann(0)): 0.020214901680800382,
                (cre(1), cre(3), ann(3), ann(1)): 0.377781823622411,
                (cre(2), cre(0), ann(0), ann(2)): 0.4754873109620925,
                (cre(2), cre(0), ann(0), ann(3)): -0.08079285521616332,
                (cre(2), cre(0), ann(1), ann(2)): -0.0915052536161256,
                (cre(2), cre(0), ann(1), ann(3)): 0.07243027144542874,
                (cre(2), cre(1), ann(0), ann(2)): -0.0915052536161256,
                (cre(2), cre(1), ann(0), ann(3)): 0.07243027144542874,
                (cre(2), cre(1), ann(1), ann(2)): 0.33840060689985296,
                (cre(2), cre(1), ann(1), ann(3)): 0.013176380212608446,
                (cre(2), cre(2), ann(2), ann(2)): 0.48216552238293964,
                (cre(2), cre(2), ann(2), ann(3)): -0.08609947118801109,
                (cre(2), cre(2), ann(3), ann(2)): -0.08609947118801109,
                (cre(2), cre(2), ann(3), ann(3)): 0.0686973464043348,
                (cre(2), cre(3), ann(2), ann(2)): -0.08609947118801109,
                (cre(2), cre(3), ann(2), ann(3)): 0.0686973464043348,
                (cre(2), cre(3), ann(3), ann(2)): 0.3317223954790056,
                (cre(2), cre(3), ann(3), ann(3)): 0.018482996184456136,
                (cre(3), cre(0), ann(0), ann(2)): -0.08079285521616332,
                (cre(3), cre(0), ann(0), ann(3)): 0.3331774960894852,
                (cre(3), cre(0), ann(1), ann(2)): 0.07243027144542874,
                (cre(3), cre(0), ann(1), ann(3)): 0.020214901680800382,
                (cre(3), cre(1), ann(0), ann(2)): 0.07243027144542874,
                (cre(3), cre(1), ann(0), ann(3)): 0.020214901680800382,
                (cre(3), cre(1), ann(1), ann(2)): 0.013176380212608446,
                (cre(3), cre(1), ann(1), ann(3)): 0.377781823622411,
                (cre(3), cre(2), ann(2), ann(2)): -0.08609947118801109,
                (cre(3), cre(2), ann(2), ann(3)): 0.3317223954790056,
                (cre(3), cre(2), ann(3), ann(2)): 0.0686973464043348,
                (cre(3), cre(2), ann(3), ann(3)): 0.018482996184456136,
                (cre(3), cre(3), ann(2), ann(2)): 0.0686973464043348,
                (cre(3), cre(3), ann(2), ann(3)): 0.018482996184456136,
                (cre(3), cre(3), ann(3), ann(2)): 0.018482996184456136,
                (cre(3), cre(3), ann(3), ann(3)): 0.3792369242328901,
            }
        )
        assert op.equiv(expected)


class TestFCIDumpErrors:
    """Covers the failure modes of :meth:`FCIDump.from_file`.

    The parser reports each of these as a recoverable error, which the binding turns into a catchable
    Python exception. That conversion is the thing under test: the underlying Rust code used to
    ``panic!``, and a panic crossing PyO3 surfaces as ``PanicException``, which derives from
    ``BaseException`` rather than ``Exception`` and so slips straight through an ordinary
    ``except Exception`` handler. Each test therefore asserts the concrete exception type *and* that
    it is an ``Exception``, since that is precisely what regressed behaviour would violate.
    """

    def test_missing_file_raises_oserror(self):
        with pytest.raises(OSError) as excinfo:
            FCIDump.from_file("does_not_exist.fcidump")
        assert isinstance(excinfo.value, Exception)
        assert "does_not_exist.fcidump" in str(excinfo.value)

    def test_missing_namelist_raises_value_error(self, tmp_path):
        # No `/` or `&END` terminator, so the header namelist is never found.
        path = tmp_path / "no_namelist.fcidump"
        path.write_text(" 0.5   1   1   1   1\n")
        with pytest.raises(ValueError, match="HEADER namelist"):
            FCIDump.from_file(str(path))

    def test_missing_norb_raises_value_error(self, tmp_path):
        path = tmp_path / "no_norb.fcidump"
        path.write_text("&FCI NELEC=   2,MS2= 0,\n /\n")
        with pytest.raises(ValueError, match="NORB"):
            FCIDump.from_file(str(path))

    def test_missing_nelec_raises_value_error(self, tmp_path):
        path = tmp_path / "no_nelec.fcidump"
        path.write_text("&FCI NORB=   2,MS2= 0,\n /\n")
        with pytest.raises(ValueError, match="NELEC"):
            FCIDump.from_file(str(path))

    def test_mo_energy_integral_raises_value_error(self, tmp_path):
        # An integral line `(i, a, j, 0)` with `i, a, j` all nonzero is an MO energy value, which the
        # parser does not support yet. Pinning it as a clean error keeps the gap visible and makes
        # any future support a deliberate change.
        path = tmp_path / "mo_energy.fcidump"
        path.write_text("&FCI NORB=   2,NELEC=   2,MS2= 0,\n /\n 0.5   1   1   2   0\n")
        with pytest.raises(ValueError, match="MO energy"):
            FCIDump.from_file(str(path))

    def test_every_failure_is_catchable_as_exception(self, tmp_path):
        """A single guard against the regression that motivated all of the above.

        ``PanicException`` would satisfy none of the ``pytest.raises`` calls above, but this states
        the property directly and independently of which exception subclass each case maps onto.
        """
        path = tmp_path / "no_norb.fcidump"
        path.write_text("&FCI NELEC=   2,MS2= 0,\n /\n")
        for bad in ("does_not_exist.fcidump", str(path)):
            try:
                FCIDump.from_file(bad)
            except Exception:
                continue
            except BaseException as exc:  # pragma: no cover - only on a regression
                raise AssertionError(
                    f"{bad!r} raised {type(exc).__name__}, which is not an `Exception` subclass and "
                    "so escapes `except Exception`"
                ) from exc
            raise AssertionError(f"{bad!r} did not raise at all")
