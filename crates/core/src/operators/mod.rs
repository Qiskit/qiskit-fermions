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
use thiserror::Error;

/// Error cases stemming from data coherence at the point of entry into `OperatorTrait` from
/// user-provided arrays.
#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("the input contains duplicate indices")]
    DuplicateIndices,
    #[error("the provided index mapping does not account for the entire length of the operator")]
    IndexMapTooSmall,
    #[error(
        "num_qubits ({num_qubits}) is too small for an operator acting on mode index {max_mode}"
    )]
    NumQubitsTooSmall { num_qubits: u32, max_mode: u32 },
}

/// Provides a total, coefficient-independent ordering key for a single term view.
///
/// Implemented by the `*TermView` structs. The key is derived purely from the term's structure
/// (the operator string it represents), so two operators that differ only in their coefficients
/// order their terms identically and scaling an operator never reshuffles it. This is what
/// [`operators::terms::ordering::canonical`](crate::operators::terms::ordering::canonical) sorts by.
pub trait TermSortKey {
    /// Returns the ordering key for this term.
    fn sort_key(&self) -> impl Ord;
}

/// Rescales the coefficient of a single term view.
///
/// Implemented by the `*TermView` structs. Because a view owns its coefficient outright and only
/// borrows the index data, rescaling one allocates nothing - it is a field update on a `Copy`
/// struct. This is what lets a weighted sum of operators be built in a single pass through
/// [`OperatorTrait::from_terms`], instead of materialising a scaled copy of every summand and then
/// concatenating those copies one at a time.
pub trait ScaledTerm {
    /// Returns this term with its coefficient multiplied by `factor`.
    fn scaled(self, factor: Complex64) -> Self;
}

pub trait OperatorTrait {
    fn zero() -> Self;
    fn one() -> Self;
    fn equiv(&self, other: &Self, atol: f64) -> bool;

    fn is_hermitian(&self, atol: f64) -> bool;

    fn adjoint(&self) -> Self;
    fn simplify(&self, atol: f64) -> Self;

    fn __iadd__(&mut self, other: &Self);
    fn __imul__(&mut self, other: Complex64);
    fn __iand__(&mut self, other: &Self);
    fn __imatmul__(&mut self, other: &Self);
    fn ichop(&mut self, atol: f64);

    /// Subtracts `other` from `self` in place.
    ///
    /// Implemented per operator type rather than as `__iadd__(&other.__neg__())` because
    /// [`__neg__`](OperatorMacro::__neg__) deep-copies *all* of `other`'s buffers - index data
    /// included - only to scale its coefficients, which `__iadd__` then copies a second time.
    /// Appending `other`'s terms while negating just the newly appended coefficients does the same
    /// work in one pass and with no temporary.
    fn __isub__(&mut self, other: &Self);

    /// Returns the composition `self & other`, i.e. with `other` applied first.
    ///
    /// This exists as its own method rather than being expressed as a clone followed by
    /// [`__iand__`](Self::__iand__) because composing rebuilds every buffer from scratch: cloning
    /// first would allocate and copy buffers that are immediately overwritten and dropped unread.
    ///
    /// The composition of two operators tracks no groups, matching `__iand__`.
    fn composed(&self, other: &Self) -> Self;

    /// Returns the composition `self @ other`, i.e. with `self` applied first.
    ///
    /// The counterpart of [`composed`](Self::composed) carrying the operand order of
    /// [`__imatmul__`](Self::__imatmul__). Both orders are spelled out once per operator type,
    /// next to the in-place operation each mirrors, rather than being derived from one another by
    /// swapping arguments at the call site.
    fn matmul(&self, other: &Self) -> Self;

    /// The borrowed view of a single term yielded by [`OperatorTrait::iter`].
    type TermView<'a>: PartialEq + TermSortKey + ScaledTerm
    where
        Self: 'a;

    /// The borrowed view of a single term (together with its group index) yielded by
    /// [`OperatorTrait::iter_with_groups`].
    ///
    /// Its [`TermSortKey`] must match that of the corresponding [`Self::TermView`] (i.e. ignore the
    /// group index), so that ordering a grouped operator agrees with ordering the same terms
    /// ungrouped.
    type GroupTermView<'a>: PartialEq + TermSortKey + ScaledTerm
    where
        Self: 'a;

    /// Iterates over the terms of the operator.
    ///
    /// The yielded `Item` is the associated type `Self::TermView<'_>` rather than a concrete view.
    /// Implementations must spell it the same way: returning the concrete view type would make the
    /// impl signature more specific than this one and trip the `refining_impl_trait` lint.
    fn iter(&self) -> impl ExactSizeIterator<Item = Self::TermView<'_>>;

    /// Returns the operator's coefficients, one per term.
    ///
    /// This is the single point of access through which the trait's coefficient-dependent provided
    /// methods ([`group_weights`](Self::group_weights)) reach the `coeffs` field, so that they need
    /// not be reimplemented per operator type.
    fn coeffs(&self) -> &[Complex64];

    /// Returns the operator's group indices, one per term, or `None` if it tracks no groups.
    ///
    /// This is the single point of access through which the trait's group-related provided methods
    /// ([`has_groups`](Self::has_groups), [`num_groups`](Self::num_groups),
    /// [`group_weights`](Self::group_weights)) reach the `groups` field, so that they need not be
    /// reimplemented per operator type.
    fn groups(&self) -> Option<&[u32]>;

    /// Returns whether the operator tracks group indices (i.e. `groups` is `Some`).
    ///
    /// Note that this is `true` even for an operator whose group indices are an empty slice, which
    /// is the state of a grouped operator with no terms (see [`num_groups`](Self::num_groups)).
    ///
    /// [`iter_with_groups`](Self::iter_with_groups) may only be called when this is `true`.
    fn has_groups(&self) -> bool {
        self.groups().is_some()
    }

    /// Returns the number of groups tracked by the operator, or `None` if it tracks no groups.
    ///
    /// The number of groups is evaluated lazily as the largest occurring group index plus 1, so it
    /// may be used as the index for the next group. An operator that tracks groups but holds no
    /// terms reports `Some(0)`.
    fn num_groups(&self) -> Option<u32> {
        let groups = self.groups()?;
        Some(match groups.iter().max() {
            Some(max) => max + 1,
            None => 0,
        })
    }

    /// Returns the mean absolute coefficient magnitude of each group, or `None` if the operator
    /// tracks no groups.
    ///
    /// The `i`-th entry is the sum of `|coeff|` over the terms in group `i`, divided by the number
    /// of terms in that group. This is the sampling weight of a randomized product formula (e.g.
    /// qDRIFT) that draws whole groups rather than individual terms.
    ///
    /// Computing this natively reduces the per-term coefficients and group indices down to one
    /// value per group in a single pass, so a caller across an FFI boundary receives only
    /// [`num_groups`](Self::num_groups) values instead of two arrays of one value per (ungrouped)
    /// term that it would have to reduce itself.
    ///
    /// A group index that no term carries yields a weight of `0.0`: the index range is dense by
    /// construction (see [`num_groups`](Self::num_groups)), but nothing enforces that, and a `0.0`
    /// weight keeps such a group out of the sample rather than poisoning every weight with a `NaN`.
    fn group_weights(&self) -> Option<Vec<f64>> {
        let groups = self.groups()?;
        let mut sums = vec![0.0; self.num_groups()? as usize];
        let mut counts = vec![0_u32; sums.len()];
        for (&group, coeff) in groups.iter().zip(self.coeffs()) {
            sums[group as usize] += coeff.norm();
            counts[group as usize] += 1;
        }
        for (sum, &count) in sums.iter_mut().zip(counts.iter()) {
            if count > 0 {
                *sum /= f64::from(count);
            }
        }
        Some(sums)
    }

    /// Iterates over the terms of the operator together with their group index.
    ///
    /// # Panics
    ///
    /// Panics if the operator does not track group indices (i.e. `groups` is `None`).
    fn iter_with_groups(&self) -> impl ExactSizeIterator<Item = Self::GroupTermView<'_>>;

    /// Constructs an operator from an iterator of term views.
    ///
    /// This is the inverse of [`OperatorTrait::iter`]: the term views may borrow from any operator
    /// (including `self`); their data is copied into the freshly-built, owned result.
    fn from_terms<'a, I>(terms: I) -> Self
    where
        Self: Sized + 'a,
        I: IntoIterator<Item = Self::TermView<'a>>;

    /// Constructs an operator (with group indices) from an iterator of group term views.
    ///
    /// This is the inverse of [`OperatorTrait::iter_with_groups`].
    fn from_terms_with_groups<'a, I>(terms: I) -> Self
    where
        Self: Sized + 'a,
        I: IntoIterator<Item = Self::GroupTermView<'a>>;

    fn get_support(&self) -> HashSet<u32>;
    fn relabel_modes(&self, permutation: Vec<u32>) -> Result<Self, CoherenceError>
    where
        Self: Sized;
}

pub trait OperatorMacro {
    fn __add__(&self, other: &Self) -> Self;
    fn __sub__(&self, other: &Self) -> Self;
    fn __mul__(&self, other: Complex64) -> Self;
    fn __div__(&self, other: Complex64) -> Self;
    fn __neg__(&self) -> Self;
    fn __and__(&self, other: &Self) -> Self;
    fn __matmul__(&self, other: &Self) -> Self;
    fn __pow__(&self, exponent: usize) -> Self;

    // more in-place operations
    fn __idiv__(&mut self, other: Complex64);
}

#[macro_export]
macro_rules! impl_operator_macro {
    ($name:ty) => {
        impl OperatorMacro for $name {
            fn __add__(&self, other: &Self) -> Self
            where
                Self: OperatorTrait,
            {
                let mut result = self.clone();
                result.__iadd__(other);
                result
            }

            fn __sub__(&self, other: &Self) -> Self
            where
                Self: OperatorTrait,
            {
                // Unlike the composing operations below, this clone is load-bearing: `__isub__`
                // appends to the existing buffers, so the result genuinely starts out as a copy of
                // `self`.
                let mut result = self.clone();
                result.__isub__(other);
                result
            }

            fn __mul__(&self, other: Complex64) -> Self
            where
                Self: OperatorTrait,
            {
                let mut result = self.clone();
                result.__imul__(other);
                result
            }

            fn __div__(&self, other: Complex64) -> Self
            where
                Self: OperatorTrait,
            {
                let mut result = self.clone();
                result.__imul__(1.0 / other);
                result
            }

            fn __idiv__(&mut self, other: Complex64)
            where
                Self: OperatorTrait,
            {
                self.__imul__(1.0 / other);
            }

            fn __neg__(&self) -> Self
            where
                Self: OperatorTrait,
            {
                self.__mul__(Complex64::new(-1.0, 0.0))
            }

            fn __and__(&self, other: &Self) -> Self
            where
                Self: OperatorTrait,
            {
                // Composing reads both operands through borrowed term views and returns freshly
                // built buffers, so there is nothing a clone of `self` could contribute here.
                self.composed(other)
            }

            fn __matmul__(&self, other: &Self) -> Self
            where
                Self: OperatorTrait,
            {
                self.matmul(other)
            }

            fn __pow__(&self, exponent: usize) -> Self
            where
                Self: OperatorTrait,
            {
                let mut result = Self::one();
                for _ in 0..exponent {
                    result.__iand__(self);
                }
                result
            }
        }

        impl Add for $name {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                self.__add__(&other)
            }
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                self.__iadd__(&other);
            }
        }

        impl Sub for $name {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                self.__sub__(&other)
            }
        }

        impl SubAssign for $name {
            fn sub_assign(&mut self, other: Self) {
                self.__isub__(&other);
            }
        }

        impl Mul<Complex64> for $name {
            type Output = Self;

            fn mul(self, other: Complex64) -> Self {
                self.__mul__(other)
            }
        }

        impl Mul<$name> for Complex64 {
            type Output = $name;

            fn mul(self, other: $name) -> $name {
                other.__mul__(self)
            }
        }

        impl MulAssign<Complex64> for $name {
            fn mul_assign(&mut self, other: Complex64) {
                self.__imul__(other);
            }
        }

        impl Div<Complex64> for $name {
            type Output = Self;

            fn div(self, other: Complex64) -> Self {
                self.__div__(other)
            }
        }

        impl DivAssign<Complex64> for $name {
            fn div_assign(&mut self, other: Complex64) {
                self.__idiv__(other);
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(self) -> Self {
                self.__neg__()
            }
        }

        impl BitAnd for $name {
            type Output = Self;

            fn bitand(self, other: Self) -> Self {
                self.__and__(&other)
            }
        }

        impl BitAndAssign for $name {
            fn bitand_assign(&mut self, other: Self) {
                self.__iand__(&other);
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::operators::edge_vertex_operator::EdgeVertexOperator;
    use crate::operators::fermion_operator::FermionOperator;
    use crate::operators::majorana_operator::MajoranaOperator;
    use crate::operators::transfer_vertex_operator::TransferVertexOperator;

    /// Calls [`OperatorTrait::is_hermitian`] through a generic bound.
    ///
    /// The bound is the point of this helper: it can only resolve if `is_hermitian` is reachable
    /// through the trait itself, so it would fail to compile if the method were merely inherent on
    /// each operator type. It also pins the trait implementation as the one that runs, which an
    /// inherent method of the same name would otherwise silently shadow at every concrete call
    /// site.
    fn is_hermitian_via_trait<T: OperatorTrait + OperatorMacro>(op: &T, atol: f64) -> bool {
        op.is_hermitian(atol)
    }

    /// Asserts the trait-level contract that holds for *every* operator type.
    ///
    /// The multiplicative identity is Hermitian and `i` times it is anti-Hermitian, both of which
    /// are expressible without knowing the term vocabulary of any specific operator type. The
    /// anti-Hermitian case is what gives this test teeth: without it, an implementation that
    /// unconditionally returned `true` would pass.
    fn assert_is_hermitian_contract<T: OperatorTrait + OperatorMacro>() {
        let one = T::one();
        assert!(is_hermitian_via_trait(&one, 1e-8));

        let imaginary = one.__mul__(Complex64::new(0.0, 1.0));
        assert!(!is_hermitian_via_trait(&imaginary, 1e-8));

        assert!(is_hermitian_via_trait(&T::zero(), 1e-8));
    }

    #[test]
    fn test_is_hermitian_through_trait_bound() {
        assert_is_hermitian_contract::<FermionOperator>();
        assert_is_hermitian_contract::<MajoranaOperator>();
        assert_is_hermitian_contract::<EdgeVertexOperator>();
        assert_is_hermitian_contract::<TransferVertexOperator>();
    }
}
