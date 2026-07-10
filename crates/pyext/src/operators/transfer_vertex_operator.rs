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

use qiskit_fermions_core::operators::transfer_vertex_operator::TransferVertexOperator;
use qiskit_fermions_core::operators::{OperatorMacro, OperatorTrait};

pub type PyTransferAction = (u32, u32);

#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.transfer_vertex_operator",
    name = "TransferVertexOperatorDataIter"
)]
struct TransferVertexOperatorDataIter {
    inner: std::vec::IntoIter<(Vec<PyTransferAction>, Complex64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TransferVertexOperatorDataIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<PyTransferAction>, Complex64)> {
        slf.inner.next()
    }
}

#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.transfer_vertex_operator",
    name = "TransferVertexOperatorDataGroupIter"
)]
struct TransferVertexOperatorDataGroupIter {
    inner: std::vec::IntoIter<(Vec<PyTransferAction>, Complex64, u32)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TransferVertexOperatorDataGroupIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<PyTransferAction>, Complex64, u32)> {
        slf.inner.next()
    }
}

/// An transfer-vertex operator.
///
/// .. _TransferVertexOperator-definition:
///
/// Definition
/// ==========
///
/// This operator is defined in terms of the transfer-vertex (:math:`T_{jk}`, :math:`V_j`)
/// operators:
///
/// .. math::
///
///     \begin{align}
///     V_j    &= -i \gamma_{2j-1} \gamma_{2j}
///             = -(a_j a_j - a_j a^\dagger_j + a^\dagger_j a_j - a^\dagger_j a^\dagger_j)
///             = 1 - 2 a^\dagger_j a_j \, , \nonumber \\
///     T_{jk} &= \frac{i}{2} V_j E_{jk}
///             = \frac{1}{2} \gamma_{2j} \gamma_{2k-1} \, , \nonumber \\
///     T_{kj} &= \frac{i}{2} E_{jk} V_k
///             = -\frac{1}{2} \gamma_{2j-1} \gamma_{2k} \nonumber
///     \end{align}
///
/// where :math:`E_{jk}` is an edge operator of the :class:`.EdgeVertexOperator` and
/// these individual terms fulfill the following mixed fermionic-bosonic commutation relations for
/// :math:`j \lt k \lt l \lt m`: [1]_
///
/// .. math::
///
///     \begin{align}
///     \left\{ T_{jk}, V_k \right\} &= 0 \nonumber \\
///     \left\{ T_{jk}, T_{lk} \right\} &= 0 \nonumber \\
///     \left[ V_k, V_l \right] &= 0 \nonumber \\
///     \left[ T_{jk}, V_l \right] &= 0 \nonumber \\
///     \left[ T_{jk}, T_{lm} \right] &= 0 \nonumber \\
///     \left[ T_{jk}, T_{kj} \right] &= 0 \nonumber \\
///     \left[ T_{jk}, T_{km} \right] &= 0 \nonumber \, .
///     \end{align}
///
/// A simple example can be represented visually like so:
///
/// .. plot::
///    :alt: A visual depication of a transfer-vertex operator.
///    :context: close-figs
///
///    >>> import rustworkx as rx
///    >>> N = 4
///    >>> graph = rx.PyDiGraph()
///    >>> _ = graph.add_nodes_from(range(N))
///    >>> _ = graph.add_edges_from([(i, i+1, i) for i in range(N-1)])
///    >>> _ = graph.add_edges_from([(i, i-1, -i) for i in range(1, N)])
///    >>> from rustworkx.visualization import mpl_draw
///    >>> mpl_draw(
///    ...     graph,
///    ...     pos={i: (i, -0.1*i) for i in range(4)},
///    ...     labels=lambda v: f"$V_{v}$",
///    ...     edge_labels=lambda e: f"$T_{{{e}{e+1}}}$" if e >= 0 else f"$T_{{{-e}{-e-1}}}$",
///    ...     with_labels=True,
///    ...     node_color="orange",
///    ... )
///    <Figure size ... with 1 Axes>
///
/// We can abuse the notation a little bit and define :math:`V_j = T_{jj}` which reflects how the
/// internal data structure of this operator works. This makes the definition of the entire
/// operator the following:
///
/// .. math::
///
///    \text{\texttt{EdgeVertexOperator}} = \sum_i c_i \bigotimes_{lr} T_{lr} \, ,
///
/// where :math:`lr` indexing the involved operator terms and :math:`c_i` is the (complex)
/// coefficient making up the linear combination of products. The indices :math:`l` and :math:`r`
/// can take any value between 0 and the number of fermionic modes acted upon by the operator minus
/// 1.
///
/// We will refer to :math:`T_{lr}` as `generalized` transfer operators.
///
/// .. _TransferVertexOperator-implementation:
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
///    ================= ========================================================================================
///    ``coeffs``        A vector of complex coefficients consisting of two 64-bit floating point numbers.
///    ``left_indices``  A vector of 32-bit integers storing the `left` fermionic mode indices (:math:`l` above).
///    ``right_indices`` A vector of 32-bit integers storing the `right` fermionic mode indices (:math:`r` above).
///    ``boundaries``    A vector of integers indicating the boundaries in ``actions`` and ``modes``.
///    ================= ========================================================================================
///
/// Fermionic modes indexed by ``left_indices`` and ``right_indices`` are considered spinless.
///
/// .. note::
///    You may access **read-only copies** of these internal arrays via their respective methods:
///    :meth:`.get_coeffs`, :meth:`.get_left_indices`, :meth:`.get_right_indices`, and
///    :meth:`.get_boundaries`.
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
///     >>> from qiskit_fermions.operators import TransferVertexOperator
///     >>> coeffs = [1.0, 2.0, -3.0, 4.0j, -0.5j]
///     >>> left_indices = [0, 3, 0, 2, 3, 0]
///     >>> right_indices = [1, 4, 1, 2, 3, 1]
///     >>> boundaries = [0, 0, 1, 2, 4, 6]
///     >>> op = TransferVertexOperator(coeffs, left_indices, right_indices, boundaries)
///     >>> print(format(op))
///       1.000000e0 +0.000000e0j * ()
///       2.000000e0 +0.000000e0j * (T(0,1))
///       0.000000e0 +4.000000e0j * (T(0,1) V(2))
///      -0.000000e0-5.000000e-1j * (V(3) T(0,1))
///      -3.000000e0 +0.000000e0j * (T(3,4))
///
/// For convenience, it is possible to construct an operator from a Python dictionary like so:
///
/// .. doctest::
///
///     >>> op = TransferVertexOperator.from_dict(
///     ...     {
///     ...         (): 1.0,
///     ...         ((0, 1),): 2.0,
///     ...         ((3, 4),): -3.0,
///     ...         ((0, 1), (2, 2)): 4.0j,
///     ...         ((3, 3), (0, 1)): -0.5j,
///     ...     }
///     ... )
///     >>> print(format(op))
///       1.000000e0 +0.000000e0j * ()
///       2.000000e0 +0.000000e0j * (T(0,1))
///       0.000000e0 +4.000000e0j * (T(0,1) V(2))
///      -0.000000e0-5.000000e-1j * (V(3) T(0,1))
///      -3.000000e0 +0.000000e0j * (T(3,4))
///
/// In addition, the following construction and quick helper methods are available:
///
/// .. autosummary::
///
///    zero
///    one
///    from_terms
///    from_terms_with_groups
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
///       2.000000e0 +0.000000e0j * (T(0,1))
///       0.000000e0 +4.000000e0j * (T(0,1) V(2))
///      -0.000000e0-5.000000e-1j * (V(3) T(0,1))
///      -3.000000e0 +0.000000e0j * (T(3,4))
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
///     TransferVertexOperator.from_dict({...})
///
/// Finally, for large operators both of these outputs may be very long and undesirable. Then, a
/// very simple form with minimal information can be obtained from the :py:func:`str` function:
///
/// .. doctest::
///
///     >>> print(str(op))
///     <TransferVertexOperator with 5 terms>
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
///     TypeError: 'qiskit_fermions.operators.transfer_vertex_operator.TransferVertexOperator' object is not iterable
///
/// Instead, this class provides custom iterators to fulfill this purpose:
///
/// .. doctest::
///
///     >>> list(sorted(op.iter_terms()))
///     [([], (1+0j)), ([(0, 1)], (2+0j)), ([(0, 1), (2, 2)], 4j), ([(3, 3), (0, 1)], (-0-0.5j)), ([(3, 4)], (-3+0j))]
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
///    iter_terms_with_groups
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
///     >>> op = TransferVertexOperator.one()
///     >>> (op + op).simplify()
///     TransferVertexOperator.from_dict({(): 2+0j})
///     >>> (op - op).simplify()
///     TransferVertexOperator.from_dict({})
///     >>> op += op
///     >>> op.simplify()
///     TransferVertexOperator.from_dict({(): 2+0j})
///     >>> op -= op
///     >>> op.simplify()
///     TransferVertexOperator.from_dict({})
///
/// Scalar Multiplication/Divison
/// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// .. doctest::
///
///     >>> op = TransferVertexOperator.one()
///     >>> (2 * op).simplify()
///     TransferVertexOperator.from_dict({(): 2+0j})
///     >>> (op / 2).simplify()
///     TransferVertexOperator.from_dict({(): 0.5+0j})
///     >>> op *= 2
///     >>> op.simplify()
///     TransferVertexOperator.from_dict({(): 2+0j})
///     >>> op /= 2
///     >>> op.simplify()
///     TransferVertexOperator.from_dict({(): 1+0j})
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
///     >>> op1 = TransferVertexOperator.from_dict({(): 2.0, ((0, 1),): 3.0})
///     >>> op2 = TransferVertexOperator.from_dict({(): 1.5, ((2, 2),): 4.0})
///     >>> comp = (op1 & op2).simplify()
///     >>> print(format(comp))
///       3.000000e0 +0.000000e0j * ()
///       4.500000e0 +0.000000e0j * (T(0,1))
///       8.000000e0 +0.000000e0j * (V(2))
///       1.200000e1 +0.000000e0j * (V(2) T(0,1))
///     >>> op2 &= op1
///     >>> print(format(op2.simplify()))
///       3.000000e0 +0.000000e0j * ()
///       4.500000e0 +0.000000e0j * (T(0,1))
///       1.200000e1 +0.000000e0j * (T(0,1) V(2))
///       8.000000e0 +0.000000e0j * (V(2))
///     >>> squared = (op1 ** 2).simplify()
///     >>> print(format(squared))
///       4.000000e0 +0.000000e0j * ()
///       1.200000e1 +0.000000e0j * (T(0,1))
///       9.000000e0 +0.000000e0j * (T(0,1) T(0,1))
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
///
/// .. [1] Gandon et al., `arXiv:2512.11418 <https://arxiv.org/abs/2512.11418v2>`_.
#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.transfer_vertex_operator",
    name = "TransferVertexOperator"
)]
#[derive(Clone)]
pub struct PyTransferVertexOperator {
    pub inner: TransferVertexOperator,
}

crate::impl_operator_magic_methods!(PyTransferVertexOperator);

#[gen_stub_pymethods]
#[pymethods]
impl PyTransferVertexOperator {
    #[new]
    fn new(
        coeffs: Vec<Complex64>,
        left_indices: Vec<u32>,
        right_indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            inner: TransferVertexOperator {
                coeffs,
                left_indices,
                right_indices,
                boundaries,
                groups: None,
            },
        }
    }

    /// Constructs a new operator from a dictionary.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict(
    ///     ...     {
    ///     ...         (): 1.0-1.0j,
    ///     ...         ((0, 0),): 2.0,
    ///     ...         ((0, 1),): 2.0j,
    ///     ...     }
    ///     ... )
    ///     >>> print(format(op))
    ///       1.000000e0 -1.000000e0j * ()
    ///       2.000000e0 +0.000000e0j * (V(0))
    ///       0.000000e0 +2.000000e0j * (T(0,1))
    ///
    /// Args:
    ///     data: a dictionary mapping tuples of terms to complex coefficients. Each key is a tuple
    ///         of ``(int, int)`` pairs indicating the indices of the generalized transfer operator,
    ///         :math:`T_{lr}` (if :math:`l = r` then this corresponds to the vertex operator
    ///         :math:`V_l`).
    ///
    /// Returns:
    ///     A new operator.
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: HashMap<Vec<(u32, u32)>, Complex64>) -> Self {
        let mut coeffs = vec![];
        let mut left_indices = vec![];
        let mut right_indices = vec![];
        let mut boundaries = vec![0];

        data.iter().for_each(|(terms, coeff)| {
            coeffs.push(*coeff);
            terms.iter().for_each(|(lidx, ridx)| {
                left_indices.push(*lidx);
                right_indices.push(*ridx);
            });
            boundaries.push(right_indices.len());
        });

        Self {
            inner: TransferVertexOperator {
                coeffs,
                left_indices,
                right_indices,
                boundaries,
                groups: None,
            },
        }
    }

    /// Returns a read-only list of the operator's coefficients.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure,
    ///    :ref:`here <TransferVertexOperator-implementation>`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.one()
    ///     >>> op += -1j * TransferVertexOperator.one()
    ///     >>> op.get_coeffs()
    ///     [(1+0j), -1j]
    ///
    /// Returns:
    ///     A list of the operator's coefficients.
    fn get_coeffs(&self) -> Vec<Complex64> {
        self.inner.coeffs().to_vec()
    }

    /// Returns a read-only list of the left indices of all generalized transfer operator terms.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure,
    ///    :ref:`here <TransferVertexOperator-implementation>`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({((0, 0),): 1.0})
    ///     >>> op += TransferVertexOperator.from_dict({((0, 1),): 1.0})
    ///     >>> op.get_left_indices()
    ///     [0, 0]
    ///
    /// Returns:
    ///     A list of the left indices of all generalized transfer operator terms.
    fn get_left_indices(&self) -> Vec<u32> {
        self.inner.left_indices().to_vec()
    }

    /// Returns a read-only list of the right indices of all generalized transfer operator terms.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure,
    ///    :ref:`here <TransferVertexOperator-implementation>`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({((0, 0),): 1.0})
    ///     >>> op += TransferVertexOperator.from_dict({((0, 1),): 1.0})
    ///     >>> op.get_right_indices()
    ///     [0, 1]
    ///
    /// Returns:
    ///     A list of the right indices of all generalized transfer operator terms.
    fn get_right_indices(&self) -> Vec<u32> {
        self.inner.right_indices().to_vec()
    }

    /// Returns a read-only list of the indices indicating the boundaries between operator terms.
    ///
    /// .. note::
    ///    This method returns a **copy** of the internal data.
    ///
    /// .. seealso::
    ///    The explanation of the internal data structure,
    ///    :ref:`here <TransferVertexOperator-implementation>`.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.one()
    ///     >>> op += TransferVertexOperator.from_dict({((0, 1),): 1.0})
    ///     >>> op.get_boundaries()
    ///     [0, 0, 1]
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict(
    ///     ...     {
    ///     ...         ((0, 1), (3, 4)): 1,
    ///     ...         ((7, 7),): 1,
    ///     ...     }
    ///     ... )
    ///     >>> assert op.get_support() == {0, 1, 3, 4, 7}
    ///
    /// Returns:
    ///     The set of mode indices which this operator acts upon.
    fn get_support(&self) -> HashSet<u32> {
        self.inner.get_support()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp, _py: Python<'_>) -> PyResult<bool> {
        match op {
            CompareOp::Eq => {
                let coeffs_eq = self.inner.coeffs == other.inner.coeffs;
                if !coeffs_eq {
                    return Ok(false);
                }
                let left_indices_eq = self.inner.left_indices == other.inner.left_indices;
                if !left_indices_eq {
                    return Ok(false);
                }
                let right_indices_eq = self.inner.right_indices == other.inner.right_indices;
                if !right_indices_eq {
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
                let left_indices_neq = self.inner.left_indices != other.inner.left_indices;
                if !left_indices_neq {
                    return Ok(false);
                }
                let right_indices_neq = self.inner.right_indices != other.inner.right_indices;
                if !right_indices_neq {
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
                .map(|(lidx, ridx)| format!("({}, {})", lidx, ridx))
                .collect();
            let key_str = if key_parts.is_empty() {
                // NOTE: we explicitly handle the zero-length case
                "()".to_string()
            } else {
                // NOTE: we explicitly enforce a final comma inside the tuple to ensure that a
                // tuple of tuples of length 1 still works as intended
                format!("({},)", key_parts.join(", "))
            };
            let val_str = format!("{}{:+}j", term.coeff.re, term.coeff.im);
            items_str.push(format!("{key_str}: {val_str}"));
        }
        Ok(format!(
            "TransferVertexOperator.from_dict({{{}}})",
            items_str.join(", ")
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "<TransferVertexOperator with {} terms>",
            self.__len__()
        ))
    }

    fn __format__(&self, _format_spec: &str) -> PyResult<String> {
        let mut sorted: Vec<_> = self.inner.iter().collect();
        sorted.sort_by_key(|&term| term.into_vec());
        let mut items_str = Vec::new();
        for term in sorted {
            let key_parts: Vec<String> = term
                .iter()
                .map(|(lidx, ridx)| {
                    if lidx == ridx {
                        format!("V({})", lidx)
                    } else {
                        format!("T({},{})", lidx, ridx)
                    }
                })
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({(): 2.0})
    ///     >>> zero = TransferVertexOperator.zero()
    ///     >>> op + zero == op
    ///     True
    ///
    /// ..
    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: TransferVertexOperator::zero(),
        }
    }

    /// Constructs the multiplicative identity operator.
    ///
    /// Composing the operator that is constructed by this method with another one has no effect.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({(): 2.0})
    ///     >>> one = TransferVertexOperator.one()
    ///     >>> op & one == op
    ///     True
    ///
    /// ..
    #[classmethod]
    fn one(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: TransferVertexOperator::one(),
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> coeffs = [1e-5] * int(1e5)
    ///     >>> boundaries = [0] + [0] * int(1e5)
    ///     >>> op = TransferVertexOperator(coeffs, [], [], boundaries)
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({(): 1e-4, ((1, 0),): 1e-6, ((0, 1),): 1e-10})
    ///     >>> print(format(op))
    ///       1.000000e-4 +0.000000e0j * ()
    ///      1.000000e-10 +0.000000e0j * (T(0,1))
    ///       1.000000e-6 +0.000000e0j * (T(1,0))
    ///     >>> op.ichop()
    ///     >>> print(format(op))
    ///       1.000000e-4 +0.000000e0j * ()
    ///       1.000000e-6 +0.000000e0j * (T(1,0))
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({(): 2.0, ((0, 0),): 1.0, ((0, 1),): -1.0j})
    ///     >>> list(sorted(op.iter_terms()))
    ///     [([], (2+0j)), ([(0, 0)], (1+0j)), ([(0, 1)], (-0-1j))]
    ///
    /// ..
    fn iter_terms(slf: PyRef<'_, Self>) -> PyResult<Py<TransferVertexOperatorDataIter>> {
        let vectorized: Vec<(Vec<PyTransferAction>, Complex64)> = slf
            .inner
            .iter()
            .map(|term| (term.into_vec(), term.coeff))
            .collect();
        let iter = TransferVertexOperatorDataIter {
            inner: vectorized.into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    /// Constructs a new operator from an iterator of terms (see also :meth:`.iter_terms`).
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({(): 2.0, ((0, 0),): 1.0, ((0, 1),): -1.0j})
    ///     >>> op.equiv(TransferVertexOperator.from_terms(op.iter_terms()))
    ///     True
    ///
    /// Args:
    ///     terms: an iterator of terms as produced by :meth:`.iter_terms`.
    ///
    /// Returns:
    ///     A new operator.
    #[classmethod]
    fn from_terms(_cls: &Bound<'_, PyType>, terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut inner = TransferVertexOperator::zero();
        terms.try_iter()?.try_for_each(|item| -> PyResult<()> {
            let (term, coeff) = item?.extract::<(Vec<PyTransferAction>, Complex64)>()?;
            inner.coeffs.push(coeff);
            term.iter().for_each(|(l, r)| {
                inner.left_indices.push(*l);
                inner.right_indices.push(*r);
            });
            inner.boundaries.push(inner.right_indices.len());
            Ok(())
        })?;
        Ok(Self { inner })
    }

    /// An iterator over the operator's terms with their associated group index.
    ///
    /// .. warning::
    ///    Mutating the iteration items does **not** affect the underlying operator data.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator([2.0, 1.0, -1.0j], [0, 0], [0, 1], [0, 0, 1, 2])
    ///     >>> op.groups = [0, 1, 1]
    ///     >>> list(op.iter_terms_with_groups())
    ///     [([], (2+0j), 0), ([(0, 0)], (1+0j), 1), ([(0, 1)], (-0-1j), 1)]
    ///
    /// ..
    fn iter_terms_with_groups(
        slf: PyRef<'_, Self>,
    ) -> PyResult<Py<TransferVertexOperatorDataGroupIter>> {
        let vectorized: Vec<(Vec<PyTransferAction>, Complex64, u32)> = slf
            .inner
            .iter_with_groups()
            .map(|term| (term.into_vec(), term.coeff, term.group))
            .collect();
        let iter = TransferVertexOperatorDataGroupIter {
            inner: vectorized.into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    /// Constructs a new operator from an iterator of terms with groups (see also
    /// :meth:`.iter_terms_with_groups`).
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator([2.0, 1.0, -1.0j], [0, 0], [0, 1], [0, 0, 1, 2])
    ///     >>> op.groups = [0, 1, 1]
    ///     >>> reconstructed = TransferVertexOperator.from_terms_with_groups(op.iter_terms_with_groups())
    ///     >>> op.equiv(reconstructed) and op.groups == reconstructed.groups
    ///     True
    ///
    /// Args:
    ///     terms: an iterator of terms as produced by :meth:`.iter_terms_with_groups`.
    ///
    /// Returns:
    ///     A new operator.
    #[classmethod]
    fn from_terms_with_groups(
        _cls: &Bound<'_, PyType>,
        terms: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let mut inner = TransferVertexOperator::zero();
        let mut groups = vec![];
        terms.try_iter()?.try_for_each(|item| -> PyResult<()> {
            let (term, coeff, group) =
                item?.extract::<(Vec<PyTransferAction>, Complex64, u32)>()?;
            inner.coeffs.push(coeff);
            term.iter().for_each(|(l, r)| {
                inner.left_indices.push(*l);
                inner.right_indices.push(*r);
            });
            inner.boundaries.push(inner.right_indices.len());
            groups.push(group);
            Ok(())
        })?;
        inner.groups = Some(groups);
        Ok(Self { inner })
    }

    /// An optional vector of `group indices` for each term.
    ///
    /// For more information refer to the :mod:`~qiskit_fermions.operators.terms.grouping` module.
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator(
    ///     ...     [1.0, 2.0, -1.0],
    ///     ...     [0, 1, 2, 3],
    ///     ...     [1, 0, 3, 2],
    ///     ...     [0, 1, 3, 4],
    ///     ... )
    ///     >>> op.groups = [0, 1, 0]
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator(
    ///     ...     [1.0, 2.0, -1.0],
    ///     ...     [0, 1, 2, 3],
    ///     ...     [1, 0, 3, 2],
    ///     ...     [0, 1, 3, 4],
    ///     ... )
    ///     >>> print(op.split_out_groups())
    ///     None
    ///     >>> op.groups = [0, 1, 0]
    ///     >>> groups = op.split_out_groups()
    ///     >>> for g in groups:
    ///     ...     print(list(sorted(g.iter_terms())))
    ///     [([(0, 1)], (1+0j)), ([(3, 2)], (-1+0j))]
    ///     [([(1, 0), (2, 3)], (2+0j))]
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
    /// The generators of this operator (the vertex and transfer operators) are individually
    /// Hermitian, so the terms themselves are unchanged by the adjoint; only the coefficients are
    /// affected:
    ///
    /// - the coefficients are complex conjugated
    ///
    /// Note that this does not make the operator self-adjoint in general: an operator with complex
    /// coefficients differs from its adjoint (as the doctest below illustrates).
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({(): -1.0j, ((0, 0), (0, 1)): 1.0})
    ///     >>> adj = op.adjoint()
    ///     >>> print(format(adj))
    ///      -0.000000e0 +1.000000e0j * ()
    ///       1.000000e0 -0.000000e0j * (V(0) T(0,1))
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
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({(): 1e-7})
    ///     >>> zero = TransferVertexOperator.zero()
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
    /// The normal order of an operator term is defined such that all vertex operators appear
    /// before all transfer operators.
    /// Within each group, the acted-upon modes are ordered lexicographically.
    ///
    /// .. note::
    ///    When a term is being reordered, the mixed commutation and anti-commutation relations
    ///    have to be taken into account. See
    ///    :ref:`here <TransferVertexOperator-definition>` for the detailed definitions.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({((0, 1), (1, 0), (1, 2), (0, 0), (2, 2)): 1})
    ///     >>> print(format(op.normal_ordered().simplify()))
    ///      -1.000000e0 -0.000000e0j * (V(0) V(2) T(0,1) T(1,0) T(1,2))
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

    /// Returns a new operator with relabeled modes.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.operators import TransferVertexOperator
    ///     >>> op = TransferVertexOperator.from_dict({
    ///     ...     ((0, 1), (2, 3)): 1,
    ///     ...     ((1, 2), (3, 0)): 1,
    ///     ... })
    ///     >>> permutation = [4, 2, 5, 3]
    ///     >>> relabeled = op.relabel_modes(permutation)
    ///     >>> print(format(relabeled))
    ///       1.000000e0 +0.000000e0j * (T(2,5) T(3,4))
    ///       1.000000e0 +0.000000e0j * (T(4,2) T(5,3))
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
pub mod transfer_vertex_operator {
    #[pymodule_export]
    use super::PyTransferVertexOperator;
}
