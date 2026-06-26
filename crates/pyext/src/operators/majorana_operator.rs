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

use std::collections::HashSet;

use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{class::basic::CompareOp, exceptions::PyNotImplementedError};
use pyo3_stub_gen::derive::*;
use std::collections::HashMap;

use qiskit_fermions_core::operators::majorana_operator::MajoranaOperator;
use qiskit_fermions_core::operators::{OperatorMacro, OperatorTrait};

pub type PyMajoranaAction = u32;

#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.majorana_operator",
    name = "MajoranaOperatorDataIter"
)]
struct MajoranaOperatorDataIter {
    inner: std::vec::IntoIter<(Vec<PyMajoranaAction>, Complex64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MajoranaOperatorDataIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<PyMajoranaAction>, Complex64)> {
        slf.inner.next()
    }
}

/// A Majorana fermion operator.
///
/// ----
///
/// Definition
/// ==========
///
/// This operator is defined by a linear combination of products of Majorana operators [1]_, which
/// can be defined in terms of the standard fermionic second-quantization creation and annihilation
/// operators (see also :py:class:`.FermionOperator`):
///
/// .. math::
///
///     \gamma = a^\dagger + a ~~\text{and}~~ \gamma' = i(a^\dagger - a)
///
/// The key property that a Majorana fermion is its own antiparticle becomes immediately apparent:
///
/// .. math::
///
///     \gamma_i = \gamma_i^\dagger ~~\text{and}~~ \gamma_i^2 = (\gamma_i^\dagger)^2 = 1
///
/// This result in the following anti-commutation relations for :math:`2n` Majorana fermions:
///
/// .. math::
///
///     \left\{\gamma_i,\gamma_j\right\} = 2\delta_{ij}
///
/// This makes the definition of the entire operator the following:
///
/// .. math::
///
///    \text{\texttt{MajoranaOperator}} = \sum_i c_i \bigotimes_j \hat{\gamma_j} \, ,
///
/// where :math:`c_i` is the (complex) coefficient making up the linear combination of products of
/// :math:`\gamma_j`. The index :math:`j` can take any value between 0 and the number of majorana
/// fermionic modes acted upon by the operator minus 1.
///
/// ----
///
/// .. _MajoranaOperator-implementation:
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
///    ``modes``      A vector of 32-bit integers storing the majorana mode indices acted upon.
///    ``boundaries`` A vector of integers indicating the boundaries in ``actions`` and ``indices``.
///    ============== =================================================================================
///
/// The integers in ``modes`` index the Majorana modes, :math:`j`. When using the convenience
/// function :py:func:`.gamma`, even (odd) indices are used for :math:`\gamma` (:math:`\gamma'`).
///
/// .. note::
///    You may access **read-only copies** of these internal arrays via their respective methods:
///    :meth:`.get_coeffs`, :meth:`.get_modes`, and :meth:`.get_boundaries`.
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
///
///     >>> from qiskit_fermions.operators import MajoranaOperator
///     >>> coeffs = [1.0, -2.0, 3.0j, -0.5j]
///     >>> modes = [0, 1, 0, 2, 0, 1, 2, 3]
///     >>> boundaries = [0, 0, 2, 4, 8]
///     >>> op = MajoranaOperator(coeffs, modes, boundaries)
///     >>> print(format(op))
///       1.000000e0 +0.000000e0j * ()
///      -2.000000e0 +0.000000e0j * (γ0 γ'0)
///      -0.000000e0-5.000000e-1j * (γ0 γ'0 γ1 γ'1)
///       0.000000e0 +3.000000e0j * (γ0 γ1)
///
/// For convenience, it is possible to construct an operator from a Python dictionary like so:
///
/// .. doctest::
///
///     >>> from qiskit_fermions.operators import gamma
///     >>> op = MajoranaOperator.from_dict(
///     ...     {
///     ...         (): 1.0,
///     ...         (gamma(0, False), gamma(0, True)): -2.0,
///     ...         (gamma(0, False), gamma(1, False)): 3.0j,
///     ...         (gamma(0, False), gamma(0, True), gamma(1, False), gamma(1, True)): -0.5j,
///     ...     }
///     ... )
///     >>> print(format(op))
///       1.000000e0 +0.000000e0j * ()
///      -2.000000e0 +0.000000e0j * (γ0 γ'0)
///      -0.000000e0-5.000000e-1j * (γ0 γ'0 γ1 γ'1)
///       0.000000e0 +3.000000e0j * (γ0 γ1)
///
/// In this example, we have leveraged :func:`.gamma` for creating the Majorana operators.
///
/// In addition, the following construction and quick helper methods are available:
///
/// .. autosummary::
///
///    zero
///    one
///    from_terms
///
/// Formatting
/// ----------
///
/// In the examples above, the constructed operators have been printed using the output from
/// :py:func:`format`, which results in a human-readable form of the operator.
///
/// .. doctest::
///
///     >>> print(format(op))
///       1.000000e0 +0.000000e0j * ()
///      -2.000000e0 +0.000000e0j * (γ0 γ'0)
///      -0.000000e0-5.000000e-1j * (γ0 γ'0 γ1 γ'1)
///       0.000000e0 +3.000000e0j * (γ0 γ1)
///
/// .. note::
///    The printing order of ``format(op)`` gets explicitly sorted before printing. As such, it
///    does not reflect the order of the terms inside the operator.
///
/// An alternative form can be obtained from the :py:func:`repr` function, which results in a
/// Python-interpretable representation. In other words, this output can readily be copied and
/// pasted into a Python shell:
///
/// .. doctest::
///
///     >>> print(repr(op))
///     MajoranaOperator.from_dict({...})
///
/// Finally, for large operators both of these outputs may be very long and undesirable. Then, a
/// very simple form with minimal information can be obtained from the :py:func:`str` function:
///
/// .. doctest::
///
///     >>> print(str(op))
///     <MajoranaOperator with 4 terms>
///
/// Iteration
/// ---------
///
/// Since the underlying data structure is implemented in Rust and has a non-trivial layout, it
/// cannot be iterated over directly:
///
/// .. doctest::
///
///     >>> list(iter(op))
///     Traceback (most recent call last):
///       ...
///     TypeError: 'qiskit_fermions.operators.majorana_operator.MajoranaOperator' object is not iterable
///
/// Instead, this class provides custom iterators to fulfill this purpose:
///
/// .. doctest::
///
///     >>> list(sorted(op.iter_terms()))
///     [([], (1+0j)), ([0, 1], (-2+0j)), ([0, 1, 2, 3], (-0-0.5j)), ([0, 2], 3j)]
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
///
///     >>> op = MajoranaOperator.one()
///     >>> (op + op).simplify()
///     MajoranaOperator.from_dict({(): 2+0j})
///     >>> (op - op).simplify()
///     MajoranaOperator.from_dict({})
///     >>> op += op
///     >>> op.simplify()
///     MajoranaOperator.from_dict({(): 2+0j})
///     >>> op -= op
///     >>> op.simplify()
///     MajoranaOperator.from_dict({})
///
/// Scalar Multiplication/Divison
/// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// .. doctest::
///
///     >>> op = MajoranaOperator.one()
///     >>> (2 * op).simplify()
///     MajoranaOperator.from_dict({(): 2+0j})
///     >>> (op / 2).simplify()
///     MajoranaOperator.from_dict({(): 0.5+0j})
///     >>> op *= 2
///     >>> op.simplify()
///     MajoranaOperator.from_dict({(): 2+0j})
///     >>> op /= 2
///     >>> op.simplify()
///     MajoranaOperator.from_dict({(): 1+0j})
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
///
///     >>> op1 = MajoranaOperator.from_dict({(): 2.0, (gamma(0, False),): 3.0})
///     >>> op2 = MajoranaOperator.from_dict({(): 1.5, (gamma(0, True),): 4.0})
///     >>> comp = (op1 & op2).simplify()
///     >>> print(format(comp))
///       3.000000e0 +0.000000e0j * ()
///       4.500000e0 +0.000000e0j * (γ0)
///       8.000000e0 +0.000000e0j * (γ'0)
///       1.200000e1 +0.000000e0j * (γ'0 γ0)
///     >>> op2 &= op1
///     >>> print(format(op2.simplify()))
///       3.000000e0 +0.000000e0j * ()
///       4.500000e0 +0.000000e0j * (γ0)
///       1.200000e1 +0.000000e0j * (γ0 γ'0)
///       8.000000e0 +0.000000e0j * (γ'0)
///     >>> squared = (op1 ** 2).simplify()
///     >>> print(format(squared))
///       4.000000e0 +0.000000e0j * ()
///       1.200000e1 +0.000000e0j * (γ0)
///       9.000000e0 +0.000000e0j * (γ0 γ0)
///
/// .. note::
///    For convenience, the right-multiplication is implemented by ``c = a @ b`` (resulting in
///    :math:`C = A B`).
///
/// .. doctest::
///
///     >>> (op1 @ op2).equiv(op2 & op1)
///     True
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
///    relabel_modes
///
/// Properties
/// ^^^^^^^^^^
///
/// Finally, various methods exist to check certain properties of an operator:
///
/// .. autosummary::
///
///    is_hermitian
///    max_rank
///    is_even
///
/// ----
///
/// .. [1] https://en.wikipedia.org/wiki/Majorana_fermion
#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.majorana_operator",
    name = "MajoranaOperator",
    mapping
)]
#[derive(Clone)]
pub struct PyMajoranaOperator {
    pub inner: MajoranaOperator,
}

crate::impl_operator_magic_methods!(PyMajoranaOperator);

#[gen_stub_pymethods]
#[pymethods]
impl PyMajoranaOperator {
    #[new]
    fn new(coeffs: Vec<Complex64>, modes: Vec<u32>, boundaries: Vec<usize>) -> Self {
        Self {
            inner: MajoranaOperator {
                coeffs,
                modes,
                boundaries,
                groups: None,
            },
        }
    }

    /// Constructs a new operator from a dictionary.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict(
    ///     ...     {
    ///     ...         (): 1.0-1.0j,
    ///     ...         (0, 1): 2.0,
    ///     ...     }
    ///     ... )
    ///     >>> print(format(op))
    ///       1.000000e0 -1.000000e0j * ()
    ///       2.000000e0 +0.000000e0j * (γ0 γ'0)
    ///
    /// Args:
    ///     data: a dictionary mapping tuples of terms to complex coefficients. Each key is a tuple
    ///         of integers, indexing the Majorana modes. You may use :func:`.gamma` to simplify
    ///         the assignment of even and odd indices to :math:`\gamma` and :math:`\gamma'`.
    ///
    /// Returns:
    ///     A new operator.
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: HashMap<Vec<u32>, Complex64>) -> Self {
        let mut coeffs = vec![];
        let mut modes = vec![];
        let mut boundaries = vec![0];

        data.iter().for_each(|(terms, coeff)| {
            coeffs.push(*coeff);
            terms.iter().for_each(|mode| {
                modes.push(*mode);
            });
            boundaries.push(modes.len());
        });

        Self {
            inner: MajoranaOperator {
                coeffs,
                modes,
                boundaries,
                groups: None,
            },
        }
    }

    /// Returns a read-only list of the operator's coefficients.
    ///
    /// .. note::
    ///    This method returns a *copy* of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure,
    ///    :ref:`here <MajoranaOperator-implementation>`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.one()
    ///     >>> op += -1j * MajoranaOperator.one()
    ///     >>> op.get_coeffs()
    ///     [(1+0j), -1j]
    ///
    /// Returns:
    ///     A list of the operator's coefficients.
    fn get_coeffs(&self) -> Vec<Complex64> {
        self.inner.coeffs().to_vec()
    }

    /// Returns a read-only list of the operator's acted-upon mode indices.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure,
    ///    :ref:`here <MajoranaOperator-implementation>`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.one()
    ///     >>> op += MajoranaOperator.from_dict({(0, 1): 1.0})
    ///     >>> op.get_modes()
    ///     [0, 1]
    ///
    /// Returns:
    ///     A list of the operator's modes.
    fn get_modes(&self) -> Vec<u32> {
        self.inner.modes().to_vec()
    }

    /// Returns a read-only list of the indices indicating the boundaries between operator terms.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure,
    ///    :ref:`here <MajoranaOperator-implementation>`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.one()
    ///     >>> op += MajoranaOperator.from_dict({(0, 1): 1.0})
    ///     >>> op.get_boundaries()
    ///     [0, 0, 2]
    ///
    /// Returns:
    ///     A list of the operator's terms boundaries.
    fn get_boundaries(&self) -> Vec<usize> {
        self.inner.boundaries().to_vec()
    }

    /// Returns the set of mode indices which this operator acts upon.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict(
    ///     ...     {
    ///     ...         (0, 4): 1,
    ///     ...         (1, 3, 4, 7): 1,
    ///     ...     }
    ///     ... )
    ///     >>> assert op.get_support() == {0, 1, 3, 4, 7}
    ///
    /// Returns:
    ///     The set of mode indices which this operator acts upon.
    fn get_support(&self) -> HashSet<u32> {
        self.inner.get_support()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp, _py: Python<'_>) -> PyResult<bool> {
        match op {
            CompareOp::Eq => {
                let coeffs_eq = self.inner.coeffs == other.inner.coeffs;
                if !coeffs_eq {
                    return Ok(false);
                }
                let modes_eq = self.inner.modes == other.inner.modes;
                if !modes_eq {
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
                let modes_neq = self.inner.modes != other.inner.modes;
                if !modes_neq {
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
            let key_parts: Vec<String> = term.iter().map(|mode| format!("{mode}")).collect();
            // TODO: find a cleaner way to handle a 1-length tuple
            let key_str = if key_parts.len() == 1 {
                format!("({},)", key_parts.join(", "))
            } else {
                format!("({})", key_parts.join(", "))
            };
            let val_str = format!("{}{:+}j", term.coeff.re, term.coeff.im);
            items_str.push(format!("{key_str}: {val_str}"));
        }
        Ok(format!(
            "MajoranaOperator.from_dict({{{}}})",
            items_str.join(", ")
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("<MajoranaOperator with {} terms>", self.__len__()))
    }

    fn __format__(&self, _format_spec: &str) -> PyResult<String> {
        let mut sorted: Vec<_> = self.inner.iter().collect();
        sorted.sort_by_key(|&term| term.into_vec());
        let mut items_str = Vec::new();
        for term in sorted {
            let key_parts: Vec<String> = term
                .iter()
                .map(|mode| format!("γ{}{}", if mode % 2 == 1 { "'" } else { "" }, mode / 2))
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
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(): 2.0})
    ///     >>> zero = MajoranaOperator.zero()
    ///     >>> op + zero == op
    ///     True
    ///
    /// ..
    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: MajoranaOperator::zero(),
        }
    }

    /// Constructs the multiplicative identity operator.
    ///
    /// Composing the operator that is constructed by this method with another one has no effect.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(): 2.0})
    ///     >>> one = MajoranaOperator.one()
    ///     >>> op & one == op
    ///     True
    ///
    /// ..
    #[classmethod]
    fn one(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: MajoranaOperator::one(),
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
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> coeffs = [1e-5] * int(1e5)
    ///     >>> boundaries = [0] + [0] * int(1e5)
    ///     >>> op = MajoranaOperator(coeffs, [], boundaries)
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
    fn simplify(&self, atol: f64) -> Self {
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
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(): 1e-4, (0,): 1e-6, (1,): 1e-10})
    ///     >>> print(format(op))
    ///       1.000000e-4 +0.000000e0j * ()
    ///       1.000000e-6 +0.000000e0j * (γ0)
    ///      1.000000e-10 +0.000000e0j * (γ'0)
    ///     >>> op.ichop()
    ///     >>> print(format(op))
    ///       1.000000e-4 +0.000000e0j * ()
    ///       1.000000e-6 +0.000000e0j * (γ0)
    ///     >>> op.ichop(1e-5)
    ///     >>> print(format(op))
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
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(): 2.0, (0,): 1.0, (1,): -1.0j})
    ///     >>> list(sorted(op.iter_terms()))
    ///     [([], (2+0j)), ([0], (1+0j)), ([1], (-0-1j))]
    ///
    /// ..
    fn iter_terms(slf: PyRef<'_, Self>) -> PyResult<Py<MajoranaOperatorDataIter>> {
        let vectorized: Vec<(Vec<PyMajoranaAction>, Complex64)> = slf
            .inner
            .iter()
            .map(|term| (term.into_vec(), term.coeff))
            .collect();
        let iter = MajoranaOperatorDataIter {
            inner: vectorized.into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    /// Constructs a new operator from an iterator of terms (see also :meth:`.iter_terms`).
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(): 2.0, (0,): 1.0, (1,): -1.0j})
    ///     >>> op.equiv(MajoranaOperator.from_terms(op.iter_terms()))
    ///     True
    ///
    /// Args:
    ///     terms: an iterator of terms as produced by :meth:`.iter_terms`.
    ///
    /// Returns:
    ///     A new operator.
    #[classmethod]
    fn from_terms(_cls: &Bound<'_, PyType>, terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut inner = MajoranaOperator::zero();
        terms.try_iter()?.try_for_each(|item| -> PyResult<()> {
            let (term, coeff) = item?.extract::<(Vec<PyMajoranaAction>, Complex64)>()?;
            inner.coeffs.push(coeff);
            inner.modes.extend_from_slice(&term);
            inner.boundaries.push(inner.modes.len());
            Ok(())
        })?;
        Ok(Self { inner })
    }

    /// An optional vector of `group indices` for each term.
    ///
    /// For more information refer to the :mod:`~qiskit_fermions.operators.grouping` module.
    #[getter]
    pub fn get_groups(&self) -> Option<Vec<u32>> {
        self.inner.groups.clone()
    }

    /// Sets the :attr:`groups` attribute.
    #[setter]
    pub fn set_groups(&mut self, groups: Option<Vec<u32>>) {
        self.inner.groups = groups;
    }

    /// Returns the number of groups.
    ///
    /// If :attr:`groups` is ``None``, this function also returns ``None``. Otherwise, it will
    /// return the number of groups which is defined to be the largest occurring group index plus
    /// 1 (which may therefore be used as the index for the next group).
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator(
    ///     ...     [1.0, 2.0, -1.0, -2.0],
    ///     ...     [0, 1, 2, 3, 1, 0, 3, 2],
    ///     ...     [0, 2, 4, 6, 8],
    ///     ... )
    ///     >>> op.groups = [0, 1, 0, 1]
    ///     >>> op.num_groups()
    ///     2
    ///
    /// Returns:
    ///     The largest group index in :attr:`groups` plus 1.
    pub fn num_groups(&self) -> Option<u32> {
        self.inner.num_groups()
    }

    /// Splits this operator into an optional list of new operators based on :attr:`groups`.
    ///
    /// If :attr:`groups` is ``None``, this function also returns ``None``. Otherwise, it will
    /// return a list of new operators that contain those terms of this operator with the
    /// corresponding `group` index.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator(
    ///     ...     [1.0, 2.0, -1.0, -2.0],
    ///     ...     [0, 1, 2, 3, 1, 0, 3, 2],
    ///     ...     [0, 2, 4, 6, 8],
    ///     ... )
    ///     >>> print(op.split_out_groups())
    ///     None
    ///     >>> op.groups = [0, 1, 0, 1]
    ///     >>> groups = op.split_out_groups()
    ///     >>> for g in groups:
    ///     ...     print(list(sorted(g.iter_terms())))
    ///     [([0, 1], (1+0j)), ([1, 0], (-1+0j))]
    ///     [([2, 3], (2+0j)), ([3, 2], (-2+0j))]
    ///
    /// Returns:
    ///     An optional vector of one new operator for each group index in :attr:`groups`.
    fn split_out_groups(slf: PyRef<'_, Self>) -> Option<Vec<Self>> {
        let groups = slf.inner.split_out_groups();
        match groups {
            None => None,
            Some(g) => {
                let mut out = Vec::with_capacity(g.len());
                g.into_iter()
                    .for_each(|group_op| out.push(Self { inner: group_op }));
                Some(out)
            }
        }
    }

    /// Returns the Hermitian conjugate (or adjoint) of this operator.
    ///
    /// This affects the terms and coefficients as follows:
    ///
    /// - the actions in each term reverse their order
    /// - the coefficients are complex conjugated
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(): -1.0j, (gamma(0, False), gamma(0, True)): 1.0})
    ///     >>> adj = op.adjoint()
    ///     >>> print(format(adj))
    ///      -0.000000e0 +1.000000e0j * ()
    ///       1.000000e0 -0.000000e0j * (γ'0 γ0)
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
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(): 1e-7})
    ///     >>> zero = MajoranaOperator.zero()
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
    /// The normal order of an operator term is defined such that all actions are ordered by
    /// lexicographically. Whether they ascend or descend depends on the value of the ``ascending``
    /// parameter. With the convention set forth by :py:func:`.gamma` to place :math:`\gamma`
    /// (:math:`\gamma'`) on even (odd) indices, this results in the following example (for the
    /// default, ``ascending=False``):
    /// ``[\gamma(1, True), \gamma(1, False), \gamma(0, True), \gamma(0, False)] = [3, 2, 1, 0]``.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(gamma(0, False), gamma(0, True), gamma(0, False)): 1})
    ///     >>> print(format(op.normal_ordered(reduce=False)))
    ///      -1.000000e0 +0.000000e0j * (γ'0 γ0 γ0)
    ///     >>> print(format(op.normal_ordered(reduce=True)))
    ///      -1.000000e0 +0.000000e0j * (γ'0)
    ///
    /// Args:
    ///     ascending: whether indices should ascend or descend.
    ///     reduce: whether to reduce each term to its minimal form by removing actions that square
    ///         to the identity. See also the example above.
    ///
    /// Returns:
    ///     An equivalent but normal-ordered operator.
    #[pyo3(signature = (ascending=false, reduce=true))]
    fn normal_ordered(&self, ascending: bool, reduce: bool) -> Self {
        Self {
            inner: self.inner.normal_ordered(ascending, reduce),
        }
    }

    /// Returns whether this operator is Hermitian.
    ///
    /// .. note::
    ///    This check is implemented using :meth:`.equiv` on the :meth:`.normal_ordered` difference
    ///    of ``self`` and its :meth:`.adjoint` and :meth:`.zero`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({
    ///     ...     (0, 1, 2, 3): 1.00001j,
    ///     ...     (3, 2, 1, 0): -1j,
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

    /// Returns the maximum rank of the terms in this operator.
    ///
    /// .. note::
    ///    The length of the longest term can depend on the operator's form which means that (for
    ///    example) operator simplification or normal-ordering can result in a different maximum
    ///    rank.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(0, 1, 2, 3): 1})
    ///     >>> op.max_rank()
    ///     4
    ///
    /// Returns:
    ///     The maximum rank of this operator.
    fn max_rank(&self) -> u32 {
        self.inner.max_rank()
    }

    /// Returns whether this operator is even.
    ///
    /// .. note::
    ///    An operator is considered even when all of its terms contain an even number of actions.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({(0, 1): 1})
    ///     >>> op.is_even()
    ///     True
    ///     >>> op = MajoranaOperator.from_dict({(0,): 1})
    ///     >>> op.is_even()
    ///     False
    ///
    /// Returns:
    ///     Whether this operator is even.
    fn is_even(&self) -> bool {
        self.inner.is_even()
    }

    /// Returns a new operator with relabeled modes.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import MajoranaOperator
    ///     >>> op = MajoranaOperator.from_dict({
    ///     ...     (0, 1): 1,
    ///     ...     (0, 1, 2, 3): 1,
    ///     ... })
    ///     >>> permutation = [5, 6, 4, 3]
    ///     >>> relabeled = op.relabel_modes(permutation)
    ///     >>> print(format(relabeled))
    ///       1.000000e0 +0.000000e0j * (γ'2 γ3)
    ///       1.000000e0 +0.000000e0j * (γ'2 γ3 γ2 γ'1)
    ///
    /// Args:
    ///     permutation: the index permutation list.
    ///
    /// Returns:
    ///     A new operator with its modes relabeled.
    fn relabel_modes(&self, permutation: Vec<u32>) -> PyResult<Self> {
        let out = self.inner.relabel_modes(permutation);
        match out {
            Ok(op) => Ok(Self { inner: op }),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }
}

#[pymodule]
pub mod majorana_operator {
    #[pymodule_export]
    use super::PyMajoranaOperator;
}
