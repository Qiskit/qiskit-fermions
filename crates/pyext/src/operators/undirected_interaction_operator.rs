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

use qiskit_fermions_core::operators::undirected_interaction_operator::UndirectedInteractionOperator;
use qiskit_fermions_core::operators::{OperatorMacro, OperatorTrait};

pub type PyUndirectedInteraction = (u32, u32);

#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.undirected_interaction_operator",
    name = "UndirectedInteractionOperatorDataIter"
)]
struct UndirectedInteractionOperatorDataIter {
    inner: std::vec::IntoIter<(Vec<PyUndirectedInteraction>, Complex64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl UndirectedInteractionOperatorDataIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<PyUndirectedInteraction>, Complex64)> {
        slf.inner.next()
    }
}

/// An undirected interaction operator.
///
/// ----
///
/// .. [1] Gandon et al., arXiv:2512.11418; https://arxiv.org/abs/2512.11418v2.
#[gen_stub_pyclass]
#[pyclass(
    module = "qiskit_fermions.operators.undirected_interaction_operator",
    name = "UndirectedInteractionOperator"
)]
#[derive(Clone)]
pub struct PyUndirectedInteractionOperator {
    pub inner: UndirectedInteractionOperator,
}

crate::impl_operator_magic_methods!(PyUndirectedInteractionOperator);

#[gen_stub_pymethods]
#[pymethods]
impl PyUndirectedInteractionOperator {
    #[new]
    fn new(
        coeffs: Vec<Complex64>,
        left_indices: Vec<u32>,
        right_indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            inner: UndirectedInteractionOperator {
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
            inner: UndirectedInteractionOperator {
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
            let key_str = format!("({})", key_parts.join(", "));
            let val_str = format!("{}{:+}j", term.coeff.re, term.coeff.im);
            items_str.push(format!("{key_str}: {val_str}"));
        }
        Ok(format!(
            "UndirectedInteractionOperator.from_dict({{{}}})",
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
            inner: UndirectedInteractionOperator::zero(),
        }
    }

    #[classmethod]
    fn one(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: UndirectedInteractionOperator::one(),
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

    fn iter_terms(slf: PyRef<'_, Self>) -> PyResult<Py<UndirectedInteractionOperatorDataIter>> {
        let vectorized: Vec<(Vec<PyUndirectedInteraction>, Complex64)> = slf
            .inner
            .iter()
            .map(|term| (term.into_vec(), term.coeff))
            .collect();
        let iter = UndirectedInteractionOperatorDataIter {
            inner: vectorized.into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    #[classmethod]
    fn from_terms(_cls: &Bound<'_, PyType>, terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut inner = UndirectedInteractionOperator::zero();
        terms.try_iter()?.try_for_each(|item| -> PyResult<()> {
            let (term, coeff) = item?.extract::<(Vec<PyUndirectedInteraction>, Complex64)>()?;
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
pub mod undirected_interaction_operator {
    #[pymodule_export]
    use super::PyUndirectedInteractionOperator;
}
