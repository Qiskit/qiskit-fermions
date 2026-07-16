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
    ($name:ty, $py_name:literal) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            fn __add__(&self, other: &Self) -> Self {
                self.inner.__add__(&other.inner).into()
            }

            fn __iadd__(&mut self, other: &Self) {
                self.inner.__iadd__(&other.inner);
            }

            fn __sub__(&self, other: &Self) -> Self {
                self.inner.__sub__(&other.inner).into()
            }

            fn __isub__(&mut self, other: &Self) {
                self.inner.__isub__(&other.inner);
            }

            fn __mul__(&self, other: Complex64) -> Self {
                self.inner.__mul__(other).into()
            }

            fn __rmul__(&self, other: Complex64) -> Self {
                self.inner.__mul__(other).into()
            }

            fn __imul__(&mut self, other: Complex64) {
                self.inner.__imul__(other);
            }

            fn __truediv__(&self, other: Complex64) -> Self {
                self.inner.__div__(other).into()
            }

            fn __itruediv__(&mut self, other: Complex64) {
                self.inner.__idiv__(other);
            }

            fn __neg__(&self) -> Self {
                self.inner.__neg__().into()
            }

            fn __and__(&self, other: &Self) -> Self {
                self.inner.__and__(&other.inner).into()
            }

            fn __iand__(&mut self, other: &Self) {
                self.inner.__iand__(&other.inner);
            }

            fn __matmul__(&self, other: &Self) -> Self {
                self.inner.__matmul__(&other.inner).into()
            }

            fn __imatmul__(&mut self, other: &Self) {
                self.inner.__imatmul__(&other.inner);
            }

            fn __len__(&self) -> usize {
                self.inner.boundaries.len() - 1
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            fn __pow__(&self, exponent: u32, modulo: Option<u32>) -> PyResult<Self> {
                match modulo {
                    Some(_) => Err(::pyo3::exceptions::PyNotImplementedError::new_err(
                        "mod argument not supported",
                    )),
                    None => Ok(self.inner.__pow__(exponent as usize).into()),
                }
            }

            fn __str__(&self) -> PyResult<String> {
                Ok(format!("<{} with {} terms>", $py_name, self.__len__()))
            }
        }
    };
}

/// Declares the two data-iterator pyclasses for an operator type.
///
/// Every operator file needs a `<Op>DataIter` and a `<Op>DataGroupIter` pyclass exposing the
/// same `__iter__`/`__next__` protocol; only the element (action) type and the registered
/// module/class names differ. The struct idents are passed explicitly (rather than derived from
/// a prefix) so they remain greppable and no `paste`-style token pasting is required.
///
/// `$module` is the *public* module path used for `#[pyclass(module = ...)]` (so runtime
/// `__module__`/`repr` stay user-facing), while `$stub_module` is the *physical* `_lib` runtime
/// path used for `#[gen_stub(module = ...)]` so the generated stub lands under
/// `python/qiskit_fermions/_lib/**` (see `tests/README.md`, "Type stubs").
///
/// `$next_repr` and `$group_next_repr` are the Python return types of the two `__next__` methods
/// (e.g. `"tuple[list[tuple[bool, int]], complex]"`), used to override what `pyo3-stub-gen`
/// generates. `__next__` returns a Rust `Option` (whose `None` PyO3 turns into `StopIteration` at
/// runtime — the iterator protocol never surfaces the `None` to Python), but the generator would
/// mechanically render that as `typing.Optional[...]`, which makes `for term in op.iter_terms()` a
/// type error for consumers. Overriding the stub to the non-`Optional` element type matches both
/// typeshed's own `Iterator.__next__ -> T` convention and the actual runtime behaviour. They are
/// passed as fully-formed literals (rather than `concat!`-ed from an element type) because the
/// `#[gen_stub(override_return_type(...))]` attribute parser expects a plain string literal.
#[macro_export]
macro_rules! declare_operator_iters {
    (
        $iter:ident,
        $group_iter:ident,
        $module:literal,
        $stub_module:literal,
        $name:literal,
        $group_name:literal,
        $action:ty,
        $next_repr:literal,
        $group_next_repr:literal $(,)?
    ) => {
        #[gen_stub_pyclass]
        #[gen_stub(module = $stub_module)]
        #[pyclass(module = $module, name = $name)]
        struct $iter {
            inner: std::vec::IntoIter<(Vec<$action>, Complex64)>,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $iter {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            #[gen_stub(override_return_type(type_repr = $next_repr))]
            fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<$action>, Complex64)> {
                slf.inner.next()
            }
        }

        #[gen_stub_pyclass]
        #[gen_stub(module = $stub_module)]
        #[pyclass(module = $module, name = $group_name)]
        struct $group_iter {
            inner: std::vec::IntoIter<(Vec<$action>, Complex64, u32)>,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $group_iter {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            #[gen_stub(override_return_type(type_repr = $group_next_repr))]
            fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<$action>, Complex64, u32)> {
                slf.inner.next()
            }
        }
    };
}

pub mod edge_vertex_operator;
pub mod fermion_operator;
pub mod library;
pub mod majorana_operator;
pub mod terms;
pub mod transfer_vertex_operator;

#[pymodule]
pub mod operators {
    #[pymodule_export]
    use super::edge_vertex_operator::edge_vertex_operator;

    #[pymodule_export]
    use super::fermion_operator::fermion_operator;

    #[pymodule_export]
    use super::majorana_operator::majorana_operator;

    #[pymodule_export]
    use super::library::operators_library;

    #[pymodule_export]
    use super::terms::operators_terms;

    #[pymodule_export]
    use super::transfer_vertex_operator::transfer_vertex_operator;
}
