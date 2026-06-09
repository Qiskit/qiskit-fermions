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

use qiskit_fermions_core::operators::directed_interaction_operator::DirectedInteractionOperator;
use qiskit_fermions_core::operators::{OperatorMacro, OperatorTrait};

pub type PyDirectedInteraction = (u32, u32);

#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.directed_interaction_operator",
    name = "DirectedInteractionOperatorDataIter"
)]
struct DirectedInteractionOperatorDataIter {
    inner: std::vec::IntoIter<(Vec<PyDirectedInteraction>, Complex64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl DirectedInteractionOperatorDataIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<PyDirectedInteraction>, Complex64)> {
        slf.inner.next()
    }
}

/// An directed interaction operator.
///
/// ----
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
/// where :math:`E_{jk}` is an edge operator of the :class:`.UndirectedInteractionOperator` and
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
///    :alt: A visual depication of a directed interaction operator.
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
/// ----
///
/// .. [1] Gandon et al., `arXiv:2512.11418 <https://arxiv.org/abs/2512.11418v2>`_.
#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.directed_interaction_operator",
    name = "DirectedInteractionOperator"
)]
#[derive(Clone)]
pub struct PyDirectedInteractionOperator {
    pub inner: DirectedInteractionOperator,
}

crate::impl_operator_magic_methods!(PyDirectedInteractionOperator);

#[gen_stub_pymethods]
#[pymethods]
impl PyDirectedInteractionOperator {
    #[new]
    fn new(
        coeffs: Vec<Complex64>,
        left_indices: Vec<u32>,
        right_indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            inner: DirectedInteractionOperator {
                coeffs,
                left_indices,
                right_indices,
                boundaries,
                groups: None,
            },
        }
    }

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
            inner: DirectedInteractionOperator {
                coeffs,
                left_indices,
                right_indices,
                boundaries,
                groups: None,
            },
        }
    }

    fn get_coeffs(&self) -> Vec<Complex64> {
        self.inner.coeffs().to_vec()
    }

    fn get_left_indices(&self) -> Vec<u32> {
        self.inner.left_indices().to_vec()
    }

    fn get_right_indices(&self) -> Vec<u32> {
        self.inner.right_indices().to_vec()
    }

    fn get_boundaries(&self) -> Vec<usize> {
        self.inner.boundaries().to_vec()
    }

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
            "DirectedInteractionOperator.from_dict({{{}}})",
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
                .map(|(lidx, ridx)| {
                    if lidx == ridx {
                        format!("V({})", lidx)
                    } else {
                        format!("E({},{})", lidx, ridx)
                    }
                })
                .collect();
            let key_str = format!("({})", key_parts.join(" "));
            let val_str = format!("{:12.6e}{:+12.6e}j", term.coeff.re, term.coeff.im);
            items_str.push(format!("{val_str} * {key_str}"));
        }
        Ok(items_str.join("\n").to_string())
    }

    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: DirectedInteractionOperator::zero(),
        }
    }

    #[classmethod]
    fn one(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: DirectedInteractionOperator::one(),
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

    #[pyo3(signature = (atol=1e-8))]
    fn simplify(&self, atol: f64) -> Self {
        Self {
            inner: self.inner.simplify(atol),
        }
    }

    #[pyo3(signature = (atol=1e-8))]
    fn ichop(&mut self, atol: f64) {
        self.inner.ichop(atol);
    }

    fn iter_terms(slf: PyRef<'_, Self>) -> PyResult<Py<DirectedInteractionOperatorDataIter>> {
        let vectorized: Vec<(Vec<PyDirectedInteraction>, Complex64)> = slf
            .inner
            .iter()
            .map(|term| (term.into_vec(), term.coeff))
            .collect();
        let iter = DirectedInteractionOperatorDataIter {
            inner: vectorized.into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    #[classmethod]
    fn from_terms(_cls: &Bound<'_, PyType>, terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut inner = DirectedInteractionOperator::zero();
        terms.try_iter()?.try_for_each(|item| -> PyResult<()> {
            let (term, coeff) = item?.extract::<(Vec<PyDirectedInteraction>, Complex64)>()?;
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

    fn adjoint(&self) -> Self {
        Self {
            inner: self.inner.adjoint(),
        }
    }

    #[pyo3(signature = (other, atol=1e-8))]
    fn equiv(&self, other: &Self, atol: f64) -> bool {
        self.inner.equiv(&other.inner, atol)
    }

    fn normal_ordered(&self) -> Self {
        Self {
            inner: self.inner.normal_ordered(),
        }
    }

    fn relabel_modes(&self, permutation: Vec<u32>) -> PyResult<Self> {
        let out = self.inner.relabel_modes(permutation);
        match out {
            Ok(op) => Ok(Self { inner: op }),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }
}

#[pymodule]
pub mod directed_interaction_operator {
    #[pymodule_export]
    use super::PyDirectedInteractionOperator;
}
