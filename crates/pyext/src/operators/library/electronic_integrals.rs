// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::operators::fermion_operator::PyFermionOperator;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::library::electronic_integrals::{From1Body, From2Body};

#[gen_stub_pymethods]
#[pymethods]
impl PyFermionOperator {
    /// Constructs an operator from spin-symmetric triangular 1-body integrals.
    ///
    /// The resulting operator is defined by
    ///
    /// .. math::
    ///
    ///     \sum_i c^\alpha_{ii} (a^\dagger_i a_i + a^\dagger_{i+n} a_{i+n}) +
    ///     \sum_{i \lt j} c^\alpha_{ij} (a^\dagger_i a_j + a^\dagger_j a_i +
    ///                                   a^\dagger_{i+n} a_{j+n} + a^\dagger_{j+n} a_{i+n})
    ///
    /// where :math:`c^\alpha` are the integral coefficients stored in ``one_body_a``, :math:`i`
    /// and :math:`j` are the indices expanded from the triangular index :math:`ij` which indexes
    /// the array, and :math:`n` is the number of orbitals, ``norb``.
    ///
    /// .. doctest::
    ///    >>> import numpy as np
    ///    >>> from qiskit_fermions.operators import FermionOperator
    ///    >>> one_body_a = np.array([1.0, 2.0, 3.0])
    ///    >>> op = FermionOperator.from_1body_tril_spin_sym(one_body_a, norb=2)
    ///    >>> print(op)
    ///      1.000000e0 +0.000000e0j * (+_0 -_0)
    ///      2.000000e0 +0.000000e0j * (+_0 -_1)
    ///      2.000000e0 +0.000000e0j * (+_1 -_0)
    ///      3.000000e0 +0.000000e0j * (+_1 -_1)
    ///      1.000000e0 +0.000000e0j * (+_2 -_2)
    ///      2.000000e0 +0.000000e0j * (+_2 -_3)
    ///      2.000000e0 +0.000000e0j * (+_3 -_2)
    ///      3.000000e0 +0.000000e0j * (+_3 -_3)
    ///
    /// Args:
    ///     one_body_a: a 1-dimensional array of length :math:`n * (n + 1) / 2` storing the 1-body
    ///         electronic integral coefficients of the :math:`\alpha`-spin species, as a flattened
    ///         triangular matrix.
    ///     norb: the number of orbitals, :math:`n`.
    ///
    /// Returns:
    ///     The 1-body component of the electronic structure Hamiltonian as defined above.
    /// ..
    #[classmethod]
    fn from_1body_tril_spin_sym(
        _cls: &Bound<'_, PyType>,
        one_body_a: PyReadonlyArray1<f64>,
        norb: u32,
    ) -> Self {
        Self {
            inner: FermionOperator::from_1body_tril_spin_sym(one_body_a.as_array(), norb),
        }
    }

    /// Constructs an operator from separate spin-species triangular 1-body integrals.
    ///
    /// The resulting operator is defined by
    ///
    /// .. math::
    ///
    ///     \sum_i c^\alpha_{ii} a^\dagger_i a_i + c^\beta_{ii} a^\dagger_{i+n} a_{i+n} +
    ///     \sum_{i \lt j} c^\alpha_{ij} (a^\dagger_i a_j + a^\dagger_j a_i) +
    ///                    c^\beta_{ij} (a^\dagger_{i+n} a_{j+n} + a^\dagger_{j+n} a_{i+n})
    ///
    /// where :math:`c^\alpha` (:math:`c^\beta`) are the integral coefficients stored in
    /// ``one_body_a`` (``one_body_b``, resp.), :math:`i` and :math:`j` are the indices expanded
    /// from the triangular index :math:`ij` which indexes the arrays, and :math:`n` is the number
    /// of orbitals, ``norb``.
    ///
    /// .. doctest::
    ///    >>> import numpy as np
    ///    >>> from qiskit_fermions.operators import FermionOperator
    ///    >>> one_body_a = np.array([1.0, 2.0, 3.0])
    ///    >>> one_body_b = np.array([-1.0, -2.0, -3.0])
    ///    >>> op = FermionOperator.from_1body_tril_spin(one_body_a, one_body_b, norb=2)
    ///    >>> print(op)
    ///      1.000000e0 +0.000000e0j * (+_0 -_0)
    ///      2.000000e0 +0.000000e0j * (+_0 -_1)
    ///      2.000000e0 +0.000000e0j * (+_1 -_0)
    ///      3.000000e0 +0.000000e0j * (+_1 -_1)
    ///     -1.000000e0 +0.000000e0j * (+_2 -_2)
    ///     -2.000000e0 +0.000000e0j * (+_2 -_3)
    ///     -2.000000e0 +0.000000e0j * (+_3 -_2)
    ///     -3.000000e0 +0.000000e0j * (+_3 -_3)
    ///
    /// Args:
    ///     one_body_a: a 1-dimensional array of length :math:`n * (n + 1) / 2` storing the 1-body
    ///         electronic integral coefficients of the :math:`\alpha`-spin species, as a flattened
    ///         triangular matrix.
    ///     one_body_b: a 1-dimensional array of length :math:`n * (n + 1) / 2` storing the 1-body
    ///         electronic integral coefficients of the :math:`\beta`-spin species, as a flattened
    ///         triangular matrix.
    ///     norb: the number of orbitals, :math:`n`.
    ///
    /// Returns:
    ///     The 1-body component of the electronic structure Hamiltonian as defined above.
    /// ..
    #[classmethod]
    fn from_1body_tril_spin(
        _cls: &Bound<'_, PyType>,
        one_body_a: PyReadonlyArray1<f64>,
        one_body_b: PyReadonlyArray1<f64>,
        norb: u32,
    ) -> Self {
        Self {
            inner: FermionOperator::from_1body_tril_spin(
                one_body_a.as_array(),
                one_body_b.as_array(),
                norb,
            ),
        }
    }

    /// Constructs an operator from spin-symmetric triangular 2-body integrals.
    ///
    /// The resulting operator is defined by
    ///
    /// .. math::
    ///
    ///     \sum_{ijkl} \frac{1}{2} c^{\alpha\alpha}_{ijkl}
    ///         \sum_{(i,j,k,l) \in \mathcal{P}(ijkl)}
    ///             (a^\dagger_i a^\dagger_k a_l a_j +
    ///              a^\dagger_{i+n} a^\dagger_k a_l a_{j+n} +
    ///              a^\dagger_i a^\dagger_{k+n} a_{l+n} a_j +
    ///              a^\dagger_{i+n} a^\dagger_{k+n} a_{l+n} a_{j+n})
    ///
    /// where :math:`c^{\alpha\alpha}` are the integral coefficients stored in ``two_body_aa``,
    /// :math:`ijkl` is the running index of the array, :math:`\mathcal{P}` generates the unique
    /// permutations of the 4-index :math:`(i,j,k,l)` (see below), and :math:`n` is the number of
    /// orbitals, ``norb``.
    ///
    /// .. note::
    ///     ``two_body_aa`` is an S8-fold symmetric array. That means, it is the flattened
    ///     lower-triangular data of a matrix of shape ``(npair, npair)``, where
    ///     ``npair = (norb * (norb + 1) // 2``. This in turn is the lower-triangular data of the
    ///     4-dimensional array of shape ``(norb, norb, norb, norb)``.
    ///     Therefore, :math:`\mathcal{P}` above expands the flattened index :math:`ijkl` into all
    ///     index permutations :math:`(i,j,k,l)` that index this 4-dimensional array.
    ///
    /// .. doctest::
    ///    >>> import numpy as np
    ///    >>> from qiskit_fermions.operators import FermionOperator
    ///    >>> two_body_aa = np.arange(1, 7, dtype=float)
    ///    >>> op = FermionOperator.from_2body_tril_spin_sym(two_body_aa, norb=2)
    ///    >>> len(op)
    ///    64
    ///
    /// Args:
    ///     two_body_aa: a 1-dimensional array of the S8-fold symmetric 2-body electronic integral
    ///         coefficients of the :math:`\alpha\alpha`-spin species, as a flattened array.
    ///     norb: the number of orbitals, :math:`n`.
    ///
    /// Returns:
    ///     The 2-body component of the electronic structure Hamiltonian as defined above.
    /// ..
    #[classmethod]
    fn from_2body_tril_spin_sym(
        _cls: &Bound<'_, PyType>,
        two_body_aa: PyReadonlyArray1<f64>,
        norb: u32,
    ) -> Self {
        Self {
            inner: FermionOperator::from_2body_tril_spin_sym(two_body_aa.as_array(), norb),
        }
    }

    /// Constructs an operator from separate spin-species triangular 2-body integrals.
    ///
    /// The resulting operator is defined by
    ///
    /// .. math::
    ///
    ///     \sum_{ijkl} \frac{1}{2}
    ///         \sum_{(i,j,k,l) \in \mathcal{P}(ijkl)}
    ///             c^{\alpha\alpha}_{ijkl} a^\dagger_i a^\dagger_k a_l a_j +
    ///             c^{\beta\beta}_{ijkl} a^\dagger_{i+n} a^\dagger_{k+n} a_{l+n} a_{j+n}
    ///     + \sum_{ijkl} \frac{1}{2}
    ///         \sum_{(i,j,k,l) \in \mathcal{P'}(ijkl)}
    ///             c^{\alpha\beta}_{ijkl} a^\dagger_{i+n} a^\dagger_k a_l a_{j+n} +
    ///             c^{\alpha\beta}_{ijkl} a^\dagger_i a^\dagger_{k+n} a_{l+n} a_j +
    ///
    /// where :math:`c^{\alpha\alpha}` (:math:`c^{\alpha\beta}`, :math:`c^{\beta\beta}`) are the
    /// integral coefficients stored in ``two_body_aa`` (``two_body_ab``, ``two_body_bb``, resp.),
    /// :math:`ijkl` is the running index of the array, :math:`\mathcal{P}` (:math:`\mathcal{P'}`)
    /// generates the unique permutations of the 4-index :math:`(i,j,k,l)` (see below), and
    /// :math:`n` is the number of orbitals, ``norb``.
    ///
    /// .. note::
    ///     ``two_body_aa`` and ``two_body_bb`` are a S8-fold symmetric arrays. That means, they
    ///     are the flattened lower-triangular data of matrices of shape ``(npair, npair)``, where
    ///     ``npair = (norb * (norb + 1) // 2``. These in turn are the lower-triangular data of the
    ///     4-dimensional arrays of shape ``(norb, norb, norb, norb)``.
    ///     Therefore, :math:`\mathcal{P}` above expands the flattened index :math:`ijkl` into all
    ///     index permutations :math:`(i,j,k,l)` that index these 4-dimensional arrays.
    ///
    ///     However, ``two_body_ab`` is only S4-fold symmetric. Thus, it contains the full data of
    ///     the ``(npair, npair)`` matrix (but still in flattened form). :math:`\mathcal{P'}`
    ///     performs the corresponding index expansion. (In the definition above, we reused the
    ///     index :math:`ijkl` as an abuse of notation.)
    ///
    /// .. doctest::
    ///    >>> import numpy as np
    ///    >>> from qiskit_fermions.operators import FermionOperator
    ///    >>> two_body_aa = np.arange(1, 7, dtype=float)
    ///    >>> two_body_ab = np.arange(11, 20, dtype=float)
    ///    >>> two_body_bb = np.arange(-1, -7, -1, dtype=float)
    ///    >>> op = FermionOperator.from_2body_tril_spin(two_body_aa, two_body_ab, two_body_bb, norb=2)
    ///    >>> len(op)
    ///    64
    ///
    /// Args:
    ///     two_body_aa: a 1-dimensional array of the S8-fold symmetric 2-body electronic integral
    ///         coefficients of the :math:`\alpha\alpha`-spin species, as a flattened array.
    ///     two_body_ab: a 1-dimensional array of the S4-fold symmetric 2-body electronic integral
    ///         coefficients of the :math:`\alpha\beta`-spin species, as a flattened array.
    ///     two_body_bb: a 1-dimensional array of the S8-fold symmetric 2-body electronic integral
    ///         coefficients of the :math:`\beta\beta`-spin species, as a flattened array.
    ///     norb: the number of orbitals, :math:`n`.
    ///
    /// Returns:
    ///     The 2-body component of the electronic structure Hamiltonian as defined above.
    /// ..
    #[classmethod]
    fn from_2body_tril_spin(
        _cls: &Bound<'_, PyType>,
        two_body_aa: PyReadonlyArray1<f64>,
        two_body_ab: PyReadonlyArray1<f64>,
        two_body_bb: PyReadonlyArray1<f64>,
        norb: u32,
    ) -> Self {
        Self {
            inner: FermionOperator::from_2body_tril_spin(
                two_body_aa.as_array(),
                two_body_ab.as_array(),
                two_body_bb.as_array(),
                norb,
            ),
        }
    }
}
