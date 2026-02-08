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

use pyo3::prelude::*;

#[macro_export]
macro_rules! impl_operator_magic_methods {
    ($name:ty) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            fn __add__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.__add__(&other.inner),
                }
            }

            fn __iadd__(&mut self, other: &Self) {
                self.inner.__iadd__(&other.inner);
            }

            fn __sub__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.__sub__(&other.inner),
                }
            }

            fn __isub__(&mut self, other: &Self) {
                self.inner.__isub__(&other.inner);
            }

            fn __mul__(&self, other: Complex64) -> Self {
                Self {
                    inner: self.inner.__mul__(other),
                }
            }

            fn __rmul__(&self, other: Complex64) -> Self {
                Self {
                    inner: self.inner.__mul__(other),
                }
            }

            fn __imul__(&mut self, other: Complex64) {
                self.inner.__imul__(other);
            }

            fn __truediv__(&self, other: Complex64) -> Self {
                Self {
                    inner: self.inner.__div__(other),
                }
            }

            fn __itruediv__(&mut self, other: Complex64) {
                self.inner.__idiv__(other);
            }

            fn __neg__(&self) -> Self {
                Self {
                    inner: self.inner.__neg__(),
                }
            }

            fn __and__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.__and__(&other.inner),
                }
            }

            fn __iand__(&mut self, other: &Self) {
                self.inner.__iand__(&other.inner);
            }
        }
    };
}

pub mod fermion_operator;
pub mod library;
pub mod majorana_operator;

#[pymodule]
pub mod operators {
    #[pymodule_export]
    use super::fermion_operator::fermion_operator;

    #[pymodule_export]
    use super::majorana_operator::majorana_operator;

    #[pymodule_export]
    use super::library::operators_library;
}
