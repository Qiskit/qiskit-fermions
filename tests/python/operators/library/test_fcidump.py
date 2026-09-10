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

import numpy as np
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

    # The exact integral values are pinned by the Rust unittests (`test_from_file` / `..._beta` in
    # `crates/core/src/operators/library/fcidump.rs`), so the literals below are copied from there
    # deliberately: there is one ground truth for the parsed data, and these tests cover what the
    # binding adds on top of it (the packing lengths, the copy semantics and the absence convention).

    def test_packed_arrays(self):
        file_path = Path(__file__).parent / "../../../h2.fcidump"
        fcidump = FCIDump.from_file(str(file_path))

        norb = fcidump.norb
        npair = norb * (norb + 1) // 2

        one_body_a = fcidump.get_one_body_tril_a()
        two_body_aa = fcidump.get_two_body_tril_aa()

        # The stub type is dimension-erased, so the shape contract only holds if asserted here.
        assert one_body_a.shape == (npair,)
        assert two_body_aa.shape == (npair * (npair + 1) // 2,)
        assert one_body_a.dtype == np.float64
        assert two_body_aa.dtype == np.float64

        np.testing.assert_allclose(
            one_body_a,
            [-1.2563390730032502, -2.3575299028703285e-16, -0.4718960072811406],
        )
        np.testing.assert_allclose(
            two_body_aa,
            [
                0.6757101548035165,
                0.0,
                0.18093119978423133,
                0.6645817302552967,
                0.0,
                0.6985737227320183,
            ],
        )

    def test_packed_arrays_beta(self):
        file_path = Path(__file__).parent / "../../../heh.fcidump"
        fcidump = FCIDump.from_file(str(file_path))

        norb = fcidump.norb
        npair = norb * (norb + 1) // 2

        np.testing.assert_allclose(
            fcidump.get_one_body_tril_b(),
            [-2.6172710340816154, 0.13523295000711089, -1.334676966650596],
        )
        np.testing.assert_allclose(
            fcidump.get_two_body_tril_bb(),
            [
                0.9643310447658793,
                -0.17219894237602218,
                0.1373946928086696,
                0.6634447909580112,
                0.03696599236891227,
                0.7584738484657803,
            ],
        )

        # Unlike its siblings, the alpha-beta block is only 4-fold symmetric, so it stores the *full*
        # (npair, npair) matrix rather than its lower triangle.
        two_body_ab = fcidump.get_two_body_tril_ab()
        assert two_body_ab.shape == (npair**2,)

        # Its row pair indexes the alpha-spin species and its column pair the beta-spin one, so it is
        # genuinely asymmetric under exchanging the two. A transposed packing would pass every
        # length check above but fail here.
        matrix = two_body_ab.reshape(npair, npair)
        assert not np.allclose(matrix, matrix.T)

    def test_restricted_file_has_no_beta_blocks(self):
        """The three beta blocks are absent together, which :attr:`is_unrestricted` reports on."""
        restricted = FCIDump.from_file(str(Path(__file__).parent / "../../../h2.fcidump"))
        assert not restricted.is_unrestricted
        assert restricted.get_one_body_tril_b() is None
        assert restricted.get_two_body_tril_ab() is None
        assert restricted.get_two_body_tril_bb() is None

        unrestricted = FCIDump.from_file(str(Path(__file__).parent / "../../../heh.fcidump"))
        assert unrestricted.is_unrestricted
        assert unrestricted.get_one_body_tril_b() is not None
        assert unrestricted.get_two_body_tril_ab() is not None
        assert unrestricted.get_two_body_tril_bb() is not None

    def test_arrays_are_copies(self):
        """Mutating a returned array must not reach the data structure behind it."""
        file_path = Path(__file__).parent / "../../../h2.fcidump"
        fcidump = FCIDump.from_file(str(file_path))

        original = fcidump.get_one_body_tril_a()[0]
        mutated = fcidump.get_one_body_tril_a()
        mutated[0] = 42.0

        assert fcidump.get_one_body_tril_a()[0] == original

    def test_constant(self):
        h2 = FCIDump.from_file(str(Path(__file__).parent / "../../../h2.fcidump"))
        assert h2.constant == 0.7199689944489797

        heh = FCIDump.from_file(str(Path(__file__).parent / "../../../heh.fcidump"))
        assert heh.constant == 1.4399379888979593

    def test_constant_is_none_when_absent(self, tmp_path):
        """A missing constant is ``None``, not ``0.0``: the two are different facts."""
        path = tmp_path / "no_constant.fcidump"
        path.write_text("&FCI NORB=   1,NELEC=   2,MS2= 0,\n /\n 0.5   1   1   0   0\n")
        assert FCIDump.from_file(str(path)).constant is None

    @pytest.mark.parametrize("fixture", ["h2.fcidump", "heh.fcidump"])
    def test_arrays_round_trip_through_the_constructors(self, fixture):
        """The accessors are the inverses of the electronic-integral constructors.

        Feeding each returned array straight back into the matching ``from_*_tril_*`` constructor must
        rebuild the same operator that :meth:`.FermionOperator.from_fcidump` produces, up to the
        constant term (which those constructors do not carry). This pins the packing end to end
        without restating any integral value.
        """
        fcidump = FCIDump.from_file(str(Path(__file__).parent / "../../.." / fixture))
        norb = fcidump.norb

        if fcidump.is_unrestricted:
            one_body = FermionOperator.from_1body_tril_spin(
                fcidump.get_one_body_tril_a(), fcidump.get_one_body_tril_b(), norb
            )
            two_body = FermionOperator.from_2body_tril_spin(
                fcidump.get_two_body_tril_aa(),
                fcidump.get_two_body_tril_ab(),
                fcidump.get_two_body_tril_bb(),
                norb,
            )
        else:
            one_body = FermionOperator.from_1body_tril_spin_sym(fcidump.get_one_body_tril_a(), norb)
            two_body = FermionOperator.from_2body_tril_spin_sym(
                fcidump.get_two_body_tril_aa(), norb
            )

        rebuilt = one_body + two_body
        expected = FermionOperator.from_fcidump(fcidump)
        # `from_fcidump` prepends the constant as the identity term; subtract it back out.
        if fcidump.constant is not None:
            expected = expected - fcidump.constant * FermionOperator.one()

        assert rebuilt.equiv(expected)


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
