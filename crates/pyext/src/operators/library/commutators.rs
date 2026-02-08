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
use crate::operators::majorana_operator::PyMajoranaOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use qiskit_fermions_core::operators::library::commutators::{
    anti_commutator, commutator, double_commutator,
};

macro_rules! impl_commutators {
    ($name:ty) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            #[staticmethod]
            pub fn _commutator_(op_a: Self, op_b: Self) -> Self {
                Self {
                    inner: commutator(&op_a.inner, &op_b.inner),
                }
            }

            #[staticmethod]
            pub fn _anti_commutator_(op_a: Self, op_b: Self) -> Self {
                Self {
                    inner: anti_commutator(&op_a.inner, &op_b.inner),
                }
            }

            #[staticmethod]
            pub fn _double_commutator_(op_a: Self, op_b: Self, op_c: Self, sign: bool) -> Self {
                Self {
                    inner: double_commutator(&op_a.inner, &op_b.inner, &op_c.inner, sign),
                }
            }
        }
    };
}

impl_commutators!(PyFermionOperator);
impl_commutators!(PyMajoranaOperator);
