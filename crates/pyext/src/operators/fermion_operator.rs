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

use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{class::basic::CompareOp, exceptions::PyNotImplementedError};
use pyo3_stub_gen::derive::*;
use std::collections::HashMap;

use qiskit_fermions_core::operators::fermion_operator::FermionOperator;
use qiskit_fermions_core::operators::{OperatorMacro, OperatorTrait};

pub type PyFermionAction = (bool, u32);

#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.fermion_operator",
    name = "FermionOperatorDataIter"
)]
struct FermionOperatorDataIter {
    inner: std::vec::IntoIter<(Vec<PyFermionAction>, Complex64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl FermionOperatorDataIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<PyFermionAction>, Complex64)> {
        slf.inner.next()
    }
}

/// A spin-less fermionic operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This operator is defined by a linear combination of products of fermionic creation and
/// annihilation operators acting on spin-less fermionic modes. That is to say, the individual
/// terms fulfill the following anti-commutation relations: [1]_
///
/// .. math::
///
///     \left\{a^\dagger_\alpha, a^\dagger_\beta\right\} =
///     \left\{a_\alpha, a_\beta\right\} = 0,~~\text{and}~~
///     \left\{a_\alpha, a^\dagger_\beta\right\} = \delta_{\alpha\beta} \, ,
///
/// where :math:`\alpha` and :math:`\beta` do not distinguish the spin species of the fermionic
/// modes they are indexing.
///
/// This makes the definition of the entire operator the following:
///
/// .. math::
///
///    \text{\texttt{FermionOperator}} = \sum_i c_i \bigotimes_j \hat{A_j} \, ,
///
/// where :math:`\hat{A_j} \in \{ a_j, a^\dagger_j \}` and :math:`c_i` is the (complex) coefficient
/// making up the linear combination of products. The index :math:`j` can take any value between 0
/// and the number of fermionic modes acted upon by the operator minus 1.
///
/// ----
///
/// Implementation
/// ==============
///
/// This class stores the terms and coefficients in multiple sparse vectors, akin to the
/// `compressed sparse row format
/// <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_
/// commonly used for sparse matrices. More concretely, a single operator contains 4 arrays:
///
/// .. table::
///
///    ============== =================================================================================
///    ``coeffs``     A vector of complex coefficients consisting of two 64-bit floating point numbers.
///    ``actions``    A vector of booleans storing the nature of the second-quantization actions.
///    ``indices``    A vector of 32-bit integers storing the fermionic mode indices acted upon.
///    ``boundaries`` A vector of integers indicating the boundaries in ``actions`` and ``indices``.
///    ============== =================================================================================
///
/// Entries in ``actions`` indicate creation (annihilation) operators by ``True`` (``False``).
/// Fermionic modes indexed by ``indices`` are considered spinless.
///
/// This data structure allows for very efficient construction and manipulation of operators.
/// However, it implies that duplicate terms may be contained in an operator at any moment.
/// These must be resolved manually through the use of :meth:`.simplify`.
///
/// Construction
/// ------------
///
/// An operator can be constructed directly by providing the arrays outlined above:
///
/// .. doctest::
///     >>> from qiskit_fermions.operators import FermionOperator
///     >>> coeffs = [1.0, 2.0, -3.0, 4.0j, -0.5j]
///     >>> actions = [True, False, False, True, True, True, False, False]
///     >>> indices = [0, 0, 0, 1, 0, 1, 2, 3]
///     >>> boundaries = [0, 0, 1, 2, 4, 8]
///     >>> op = FermionOperator(coeffs, actions, indices, boundaries)
///     >>> print(op)
///       1.000000e0 +0.000000e0j * ()
///      -3.000000e0 +0.000000e0j * (-_0)
///       0.000000e0 +4.000000e0j * (-_0 +_1)
///       2.000000e0 +0.000000e0j * (+_0)
///      -0.000000e0-5.000000e-1j * (+_0 +_1 -_2 -_3)
///
/// For convenience, it is possible to construct an operator from a Python dictionary like so:
///
/// .. doctest::
///     >>> from qiskit_fermions.operators import cre, ann
///     >>> op = FermionOperator.from_dict(
///     ...     {
///     ...         (): 1.0,
///     ...         (cre(0),): 2.0,
///     ...         (ann(0),): -3.0,
///     ...         (ann(0), cre(1)): 4.0j,
///     ...         (cre(0), cre(1), ann(2), ann(3)): -0.5j,
///     ...     }
///     ... )
///     >>> print(op)
///       1.000000e0 +0.000000e0j * ()
///      -3.000000e0 +0.000000e0j * (-_0)
///       0.000000e0 +4.000000e0j * (-_0 +_1)
///       2.000000e0 +0.000000e0j * (+_0)
///      -0.000000e0-5.000000e-1j * (+_0 +_1 -_2 -_3)
///
/// In this example, we have leveraged :func:`.cre` and :func:`.ann` for creating the creation and
/// annihilation operators at the specified indices.
///
/// In addition, the following construction and quick helper methods are available:
///
/// .. autosummary::
///
///    zero
///    one
///
/// Iteration
/// ---------
///
/// Since the underlying data structure is implemented in Rust and has a non-trivial layout, it
/// cannot be iterated over directly:
///
/// .. doctest::
///     >>> list(iter(op))
///     Traceback (most recent call last):
///       ...
///     TypeError: 'qiskit_fermions.operators.fermion_operator.FermionOperator' object is not iterable
///
/// Instead, this class provides custom iterators to fulfill this purpose:
///
/// .. doctest::
///     >>> list(sorted(op.iter_terms()))
///     [([], (1+0j)), ([(False, 0)], (-3+0j)), ([(False, 0), (True, 1)], 4j), ([(True, 0)], (2+0j)), ([(True, 0), (True, 1), (False, 2), (False, 3)], (-0-0.5j))]
///
/// See also:
///     :meth:`iter_terms`
///         For more relevant implementation details.
///
/// The table below lists all available iterators:
///
/// .. autosummary::
///
///    iter_terms
///
/// Arithmetics
/// -----------
///
/// The following arithmetic operations are supported:
///
/// Addition/Subtraction
/// ^^^^^^^^^^^^^^^^^^^^
///
/// .. doctest::
///     >>> op = FermionOperator.one()
///     >>> (op + op).simplify()
///     FermionOperator.from_dict({(): 2+0j})
///     >>> (op - op).simplify()
///     FermionOperator.from_dict({})
///     >>> op += op
///     >>> op.simplify()
///     FermionOperator.from_dict({(): 2+0j})
///     >>> op -= op
///     >>> op.simplify()
///     FermionOperator.from_dict({})
///
/// Scalar Multiplication/Divison
/// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// .. doctest::
///     >>> op = FermionOperator.one()
///     >>> (2 * op).simplify()
///     FermionOperator.from_dict({(): 2+0j})
///     >>> (op / 2).simplify()
///     FermionOperator.from_dict({(): 0.5+0j})
///     >>> op *= 2
///     >>> op.simplify()
///     FermionOperator.from_dict({(): 2+0j})
///     >>> op /= 2
///     >>> op.simplify()
///     FermionOperator.from_dict({(): 1+0j})
///
/// Operator Composition
/// ^^^^^^^^^^^^^^^^^^^^
///
/// .. note::
///    Operator composition corresponds to left-multiplication: ``c = a & b`` corresponds to
///    :math:`C = B A`. In other words, the composition of two operators returns a resulting
///    operator that performs "first ``a`` and then ``b``".
///
/// .. doctest::
///     >>> op1 = FermionOperator.from_dict({(): 2.0, (cre(0),): 3.0})
///     >>> op2 = FermionOperator.from_dict({(): 1.5, (ann(1),): 4.0})
///     >>> comp = (op1 & op2).simplify()
///     >>> print(comp)
///       3.000000e0 +0.000000e0j * ()
///       8.000000e0 +0.000000e0j * (-_1)
///       1.200000e1 +0.000000e0j * (-_1 +_0)
///       4.500000e0 +0.000000e0j * (+_0)
///     >>> op2 &= op1
///     >>> print(op2.simplify())
///       3.000000e0 +0.000000e0j * ()
///       8.000000e0 +0.000000e0j * (-_1)
///       4.500000e0 +0.000000e0j * (+_0)
///       1.200000e1 +0.000000e0j * (+_0 -_1)
///     >>> squared = (op1 ** 2).simplify()
///     >>> print(squared)
///       4.000000e0 +0.000000e0j * ()
///       1.200000e1 +0.000000e0j * (+_0)
///       9.000000e0 +0.000000e0j * (+_0 +_0)
///
/// Other Operations
/// ^^^^^^^^^^^^^^^^
///
/// In addition to the magic methods that correspond to the arithmetic operations outlined above,
/// the following methods are available:
///
/// .. autosummary::
///
///    adjoint
///    ichop
///    simplify
///    normal_ordered
///
/// Properties
/// ^^^^^^^^^^
///
/// Finally, various methods exist to check certain properties of an operator:
///
/// .. autosummary::
///
///    is_hermitian
///    many_body_order
///    conserves_particle_number
///
/// ----
///
/// .. [1] https://en.wikipedia.org/wiki/Second_quantization#Fermion_creation_and_annihilation_operators
#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.fermion_operator",
    name = "FermionOperator"
)]
#[derive(Clone)]
pub struct PyFermionOperator {
    pub inner: FermionOperator,
}

crate::impl_operator_magic_methods!(PyFermionOperator);

#[gen_stub_pymethods]
#[pymethods]
impl PyFermionOperator {
    #[new]
    fn new(
        coeffs: Vec<Complex64>,
        actions: Vec<bool>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            inner: FermionOperator {
                coeffs,
                actions,
                indices,
                boundaries,
            },
        }
    }

    /// Constructs a new operator from a dictionary.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict(
    ///     ...     {
    ///     ...         (): 1.0-1.0j,
    ///     ...         ((True, 0), (False, 1)): 2.0,
    ///     ...     }
    ///     ... )
    ///     >>> print(op)
    ///       1.000000e0 -1.000000e0j * ()
    ///       2.000000e0 +0.000000e0j * (+_0 -_1)
    ///
    /// Args:
    ///     data: a dictionary mapping tuples of terms to complex coefficients. Each key is a tuple
    ///         of ``(bool, int)`` pairs. You may use :func:`.cre` and :func:`.ann` to simplify
    ///         their construction.
    ///
    /// Returns:
    ///     A new operator.
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: HashMap<Vec<(bool, u32)>, Complex64>) -> Self {
        let mut coeffs = vec![];
        let mut actions = vec![];
        let mut indices = vec![];
        let mut boundaries = vec![0];

        data.iter().for_each(|(terms, coeff)| {
            coeffs.push(*coeff);
            terms.iter().for_each(|(action, idx)| {
                actions.push(*action);
                indices.push(*idx);
            });
            boundaries.push(indices.len());
        });

        Self {
            inner: FermionOperator {
                coeffs,
                actions,
                indices,
                boundaries,
            },
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp, _py: Python<'_>) -> PyResult<bool> {
        match op {
            CompareOp::Eq => {
                let coeffs_eq = self.inner.coeffs == other.inner.coeffs;
                if !coeffs_eq {
                    return Ok(false);
                }
                let actions_eq = self.inner.actions == other.inner.actions;
                if !actions_eq {
                    return Ok(false);
                }
                let indices_eq = self.inner.indices == other.inner.indices;
                if !indices_eq {
                    return Ok(false);
                }
                let boundaries_eq = self.inner.boundaries == other.inner.boundaries;
                if !boundaries_eq {
                    return Ok(false);
                }
                Ok(true)
            }
            CompareOp::Ne => {
                let coeffs_neq = self.inner.coeffs != other.inner.coeffs;
                if !coeffs_neq {
                    return Ok(false);
                }
                let actions_neq = self.inner.actions != other.inner.actions;
                if !actions_neq {
                    return Ok(false);
                }
                let indices_neq = self.inner.indices != other.inner.indices;
                if !indices_neq {
                    return Ok(false);
                }
                let boundaries_neq = self.inner.boundaries != other.inner.boundaries;
                if !boundaries_neq {
                    return Ok(false);
                }
                Ok(true)
            }
            _ => Err(PyErr::new::<PyNotImplementedError, _>("")),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut items_str = Vec::new();
        for term in self.inner.iter() {
            let key_parts: Vec<String> = term
                .iter()
                .map(|(action, orb)| {
                    format!("({}, {})", if *action { "True" } else { "False" }, orb)
                })
                .collect();
            let key_str = format!("({})", key_parts.join(", "));
            let val_str = format!("{}{:+}j", term.coeff.re, term.coeff.im);
            items_str.push(format!("{key_str}: {val_str}"));
        }
        Ok(format!(
            "FermionOperator.from_dict({{{}}})",
            items_str.join(", ")
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let mut sorted: Vec<_> = self.inner.iter().collect();
        sorted.sort_by_key(|&term| term.into_vec());
        let mut items_str = Vec::new();
        for term in sorted {
            let key_parts: Vec<String> = term
                .iter()
                .map(|(action, orb)| format!("{}_{}", if *action { "+" } else { "-" }, orb))
                .collect();
            let key_str = format!("({})", key_parts.join(" "));
            let val_str = format!("{:12.6e}{:+12.6e}j", term.coeff.re, term.coeff.im);
            items_str.push(format!("{val_str} * {key_str}"));
        }
        Ok(items_str.join("\n").to_string())
    }

    /// Constructs the additive identity operator.
    ///
    /// Adding the operator that is constructed by this method to another one has no effect.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({(): 2.0})
    ///     >>> zero = FermionOperator.zero()
    ///     >>> op + zero == op
    ///     True
    ///
    /// ..
    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: FermionOperator::zero(),
        }
    }

    /// Constructs the multiplicative identity operator.
    ///
    /// Composing the operator that is constructed by this method with another one has no effect.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({(): 2.0})
    ///     >>> one = FermionOperator.one()
    ///     >>> op & one == op
    ///     True
    ///
    /// ..
    #[classmethod]
    fn one(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: FermionOperator::one(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.boundaries.len() - 1
    }

    fn __pow__(&self, exponent: u32, modulo: Option<u32>) -> PyResult<Self> {
        match modulo {
            Some(_) => Err(PyNotImplementedError::new_err("mod argument not supported")),
            None => {
                let result = Self {
                    inner: self.inner.__pow__(exponent as usize),
                };
                Ok(result)
            }
        }
    }

    /// Returns an equivalent but simplified operator.
    ///
    /// The simplification process first sums all coefficients that belong to equal terms and then
    /// only retains those whose total coefficient exceeds the specified tolerance (just like
    /// :meth:`.ichop`).
    ///
    /// When an operator has been arithmetically manipulated or constructed in a way that does not
    /// guarantee unique terms, this method should be called before applying any method that
    /// filters numerically small coefficients to avoid loss of information. See the example below
    /// which showcases how :meth:`.ichop` can truncate terms that sum to a total coefficient
    /// magnitude which should not be truncated:
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> coeffs = [1e-5] * int(1e5)
    ///     >>> boundaries = [0] + [0] * int(1e5)
    ///     >>> op = FermionOperator(coeffs, [], [], boundaries)
    ///     >>> canon = op.simplify(1e-4)
    ///     >>> assert canon.equiv(op.one(), 1e-6)
    ///     >>> op.ichop(1e-4)
    ///     >>> assert op.equiv(op.zero(), 1e-6)
    ///
    /// Args:
    ///     atol: the absolute tolerance for the cutoff. This value defaults to ``1e-8``.
    ///
    /// Returns:
    ///     An equivalent but simplified operator.
    #[pyo3(signature = (atol=1e-8))]
    fn simplify(&mut self, atol: f64) -> Self {
        Self {
            inner: self.inner.simplify(atol),
        }
    }

    /// Removes terms whose coefficient magnitude lies below the provided threshold.
    ///
    /// .. caution::
    ///    This method truncates coefficients greedily! If the acted upon operator may contain
    ///    separate coefficients for duplicate terms consider calling :meth:`.simplify` instead!
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({(): 1e-4, ((True, 0),): 1e-6, ((False, 0),): 1e-10})
    ///     >>> print(op)  # doctest: +FLOAT_CMP
    ///       1.000000e-4 +0.000000e0j * ()
    ///      1.000000e-10 +0.000000e0j * (-_0)
    ///       1.000000e-6 +0.000000e0j * (+_0)
    ///     >>> op.ichop()
    ///     >>> print(op)  # doctest: +FLOAT_CMP
    ///       1.000000e-4 +0.000000e0j * ()
    ///       1.000000e-6 +0.000000e0j * (+_0)
    ///     >>> op.ichop(1e-5)
    ///     >>> print(op)  # doctest: +FLOAT_CMP
    ///       1.000000e-4 +0.000000e0j * ()
    ///
    /// Args:
    ///     atol: the absolute tolerance for the cutoff. This value defaults to ``1e-8``.
    #[pyo3(signature = (atol=1e-8))]
    fn ichop(&mut self, atol: f64) {
        self.inner.ichop(atol);
    }

    /// An iterator over the operator's terms.
    ///
    /// .. warning::
    ///    Mutating the iteration items does **not** affect the underlying operator data.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({(): 2.0, ((True, 0),): 1.0, ((False, 1),): -1.0j})
    ///     >>> list(sorted(op.iter_terms()))
    ///     [([], (2+0j)), ([(False, 1)], (-0-1j)), ([(True, 0)], (1+0j))]
    ///
    /// ..
    fn iter_terms(slf: PyRef<'_, Self>) -> PyResult<Py<FermionOperatorDataIter>> {
        let vectorized: Vec<(Vec<PyFermionAction>, Complex64)> = slf
            .inner
            .iter()
            .map(|term| (term.into_vec(), term.coeff))
            .collect();
        let iter = FermionOperatorDataIter {
            inner: vectorized.into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    /// Returns the Hermitian conjugate (or adjoint) of this operator.
    ///
    /// This affects the terms and coefficients as follows:
    ///
    /// - the actions in each term reverse their order and flip between creation and annihilation
    /// - the coefficients are complex conjugated
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({(): -1.0j, ((True, 0), (False, 1)): 1.0})
    ///     >>> adj = op.adjoint()
    ///     >>> print(adj)  # doctest: +FLOAT_CMP
    ///      -0.000000e0 +1.000000e0j * ()
    ///       1.000000e0 -0.000000e0j * (+_1 -_0)
    ///
    /// ..
    fn adjoint(&self) -> Self {
        Self {
            inner: self.inner.adjoint(),
        }
    }

    /// Checks this operator for equivalence with another operator.
    ///
    /// Equivalence in this context means approximate equality up to the specified absolute
    /// tolerance. To be more precise, this method returns ``True``, when all the absolute values
    /// of the coefficients in the difference ``other - self`` are below the specified threshold
    /// ``atol``.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({(): 1e-7})
    ///     >>> zero = FermionOperator.zero()
    ///     >>> op.equiv(zero)
    ///     False
    ///     >>> op.equiv(zero, 1e-6)
    ///     True
    ///     >>> op.equiv(zero, 1e-9)
    ///     False
    ///
    /// Args:
    ///     other: the other operator to compare with.
    ///     atol: the absolute tolerance for the comparison. This value defaults to ``1e-8``.
    #[pyo3(signature = (other, atol=1e-8))]
    fn equiv(&self, other: &Self, atol: f64) -> bool {
        self.inner.equiv(&other.inner, atol)
    }

    /// Returns an equivalent operator with normal ordered terms.
    ///
    /// The normal order of an operator term is defined such that all creation actions before all
    /// annihilation actions and the indices of actions within each group descend lexicographically
    /// (e.g. ``+_1 +_0 -_1 -_0``).
    ///
    /// .. note::
    ///    When a term is being reordered, the anti-commutation relations have to be taken into
    ///    account, :math:`a_i a^\dagger_j = \delta_{ij} - a^\dagger_j a^i`, implying that the
    ///    number of terms may change.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({((False, 1), (True, 1), (False, 0), (True, 0)): 1})
    ///     >>> print(op.normal_ordered().simplify())  # doctest: +FLOAT_CMP
    ///       1.000000e0 +0.000000e0j * ()
    ///      -1.000000e0 +0.000000e0j * (+_0 -_0)
    ///      -1.000000e0 +0.000000e0j * (+_1 -_1)
    ///      -1.000000e0 +0.000000e0j * (+_1 +_0 -_1 -_0)
    ///
    /// Returns:
    ///     An equivalent but normal-ordered operator.
    fn normal_ordered(&self) -> Self {
        Self {
            inner: self.inner.normal_ordered(),
        }
    }

    /// Returns whether this operator is Hermitian.
    ///
    /// .. note::
    ///    This check is implemented using :meth:`.equiv` on the :meth:`.normal_ordered` difference
    ///    of ``self`` and its :meth:`.adjoint` and :meth:`.zero`.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({
    ///     ...     ((True, 0), (False, 1)): 1.00001j,
    ///     ...     ((True, 1), (False, 0)): -1j,
    ///     ... })
    ///     >>> op.is_hermitian()
    ///     False
    ///     >>> op.is_hermitian(1e-4)
    ///     True
    ///
    /// Args:
    ///     atol: The numerical accuracy upto which coefficients are considered equal. This value
    ///         defaults to ``1e-8``.
    ///
    /// Returns:
    ///     Whether this operator is Hermitian.
    #[pyo3(signature = (atol=1e-8))]
    fn is_hermitian(&self, atol: f64) -> bool {
        self.inner.is_hermitian(atol)
    }

    /// Returns the many-body order of this operator.
    ///
    /// .. note::
    ///    The many-body order is defined as the length of the longest term contained in the
    ///    operator.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({
    ///     ...     ((True, 0), (False, 1), (True, 2), (False, 3)): 1,
    ///     ... })
    ///     >>> op.many_body_order()
    ///     4
    ///
    /// Returns:
    ///     The many-body order of this operator.
    fn many_body_order(&self) -> u32 {
        self.inner.many_body_order()
    }

    /// Returns whether this operator is particle-number conserving.
    ///
    /// .. doctest::
    ///     >>> from qiskit_fermions.operators import FermionOperator
    ///     >>> op = FermionOperator.from_dict({((True, 0), (False, 1)): 1})
    ///     >>> op.conserves_particle_number()
    ///     True
    ///     >>> op = FermionOperator.from_dict({((True, 0),): 1})
    ///     >>> op.conserves_particle_number()
    ///     False
    ///
    /// Returns:
    ///     Whether this operator is particle-number conserving.
    fn conserves_particle_number(&self) -> bool {
        self.inner.conserves_particle_number()
    }
}

#[pymodule]
pub mod fermion_operator {
    #[pymodule_export]
    use super::PyFermionOperator;
}
