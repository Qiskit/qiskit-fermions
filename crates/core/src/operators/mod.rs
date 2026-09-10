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
    #[error("expected one group index per term, but got {num_groups} for {num_terms} terms")]
    GroupLengthMismatch { num_groups: usize, num_terms: usize },
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

/// Exposes the group index of a single grouped term view.
///
/// Implemented by the `*GroupTermView` structs. This is what lets the group-related routines in
/// [`operators::terms::grouping::analysis`](crate::operators::terms::grouping::analysis) bucket the
/// terms of an arbitrary operator type by group in a single pass: the group index is otherwise
/// reachable only on the concrete view structs, not through
/// [`OperatorTrait::GroupTermView`]'s bounds.
///
/// Deliberately kept out of [`TermSortKey::sort_key`], which must keep ignoring the group index so
/// that ordering a grouped operator agrees with ordering the same terms ungrouped.
pub trait GroupedTerm {
    /// Returns the index of the group this term belongs to.
    fn group(&self) -> u32;
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

    /// Adds `factor * other` to `self` in place.
    ///
    /// Implemented per operator type rather than as `__iadd__(&other.__mul__(factor))` because
    /// [`__mul__`](OperatorMacro::__mul__) deep-copies *all* of `other`'s buffers - index data
    /// included - only to scale its coefficients, which `__iadd__` then copies a second time.
    /// Appending `other`'s terms while scaling just the newly appended coefficients does the same
    /// work in one pass and with no temporary.
    ///
    /// Like [`__iadd__`](Self::__iadd__), this changes the number of terms and therefore clears the
    /// group indices.
    fn __iscaled_add__(&mut self, other: &Self, factor: Complex64);

    /// Subtracts `other` from `self` in place.
    ///
    /// The negation is folded into the coefficient scaling of
    /// [`__iscaled_add__`](Self::__iscaled_add__), so this too appends in a single pass without
    /// building a negated copy of `other` first.
    fn __isub__(&mut self, other: &Self) {
        self.__iscaled_add__(other, Complex64::new(-1.0, 0.0));
    }

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
    type GroupTermView<'a>: PartialEq + TermSortKey + ScaledTerm + GroupedTerm
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
    /// This is also the single point of access through which the group-analysis routines in
    /// [`operators::terms::grouping::analysis`](crate::operators::terms::grouping::analysis) read
    /// the coefficients, so that they need not be reimplemented per operator type.
    fn coeffs(&self) -> &[Complex64];

    /// Returns the operator's group indices, one per term, or `None` if it tracks no groups.
    ///
    /// This is the single point of access through which the trait's group-related provided methods
    /// ([`has_groups`](Self::has_groups), [`num_groups`](Self::num_groups)) and the group-analysis
    /// routines in
    /// [`operators::terms::grouping::analysis`](crate::operators::terms::grouping::analysis) reach
    /// the `groups` field, so that they need not be reimplemented per operator type.
    fn groups(&self) -> Option<&[u32]>;

    /// Overwrites the operator's group indices without checking their length.
    ///
    /// The single point at which the `groups` field is written, which is what makes
    /// `rg '\.groups = ' crates/core/src/operators/*_operator.rs` a complete review check: any hit
    /// outside the one-line implementations of this method - and the struct literals that build a
    /// whole operator at once, which have no window in which a mismatched value is observable - is a
    /// bypass of [`set_groups`](Self::set_groups)' length check.
    ///
    /// Underscore-prefixed and internal by convention, in the same sense as the operators'
    /// `_append_term`. The only callers permitted to skip the length check are
    /// [`clear_groups`](Self::clear_groups), for which `None` is unconditionally valid, and the
    /// incremental builders in
    /// [`operators::library`](crate::operators::library), which are described under
    /// [`set_groups`](Self::set_groups).
    fn _write_groups(&mut self, groups: Option<Vec<u32>>);

    /// Assigns the operator's group indices, or clears them when given `None`.
    ///
    /// This is the only route through which a *complete* array of group indices enters an operator,
    /// and it rejects one whose length differs from the number of terms. Nothing downstream
    /// re-checks that invariant: [`iter_with_groups`](Self::iter_with_groups) zips the two arrays and
    /// so would silently drop trailing terms, while [`num_groups`](Self::num_groups) reads the group
    /// indices alone and so would report groups that no term carries. Rejecting the assignment keeps
    /// both unreachable rather than leaving a corrupt operator for a later routine to detect.
    ///
    /// The incremental builders in [`operators::library`](crate::operators::library) are the
    /// deliberate exception: they seed an empty array on a term-less operator and then append terms
    /// and group indices in lockstep, so the whole-array check this performs would only ever hold in
    /// that initial empty instant. Every other write goes through here or
    /// [`clear_groups`](Self::clear_groups).
    fn set_groups(&mut self, groups: Option<Vec<u32>>) -> Result<(), CoherenceError> {
        if let Some(groups) = &groups
            && groups.len() != self.coeffs().len()
        {
            return Err(CoherenceError::GroupLengthMismatch {
                num_groups: groups.len(),
                num_terms: self.coeffs().len(),
            });
        }
        self._write_groups(groups);
        Ok(())
    }

    /// Clears the operator's group indices.
    ///
    /// The infallible counterpart of [`set_groups`](Self::set_groups): dropping a grouping can never
    /// break the one-index-per-term invariant. Spelled as its own method rather than as
    /// `set_groups(None)` so that the call sites that drop a grouping - [`__iadd__`](Self::__iadd__),
    /// [`__isub__`](Self::__isub__), [`ichop`](Self::ichop), and the C API's `del_groups` and
    /// `add_term` - need neither an `expect` nor a discarded `Result` for an outcome that cannot
    /// fail, which is also what lets those C functions keep their `void` signature.
    ///
    /// Every operation that changes the number of terms without maintaining the group indices in
    /// lockstep must call this.
    fn clear_groups(&mut self) {
        self._write_groups(None);
    }

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
    /// Returns `self + factor * other`.
    ///
    /// The fused counterpart of `self.__add__(&other.__mul__(factor))`, which would materialize a
    /// fully scaled copy of `other` just to append it. Both `__add__` and `__sub__` are the special
    /// cases `factor = 1` and `factor = -1`.
    fn __scaled_add__(&self, other: &Self, factor: Complex64) -> Self;
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

            fn __scaled_add__(&self, other: &Self, factor: Complex64) -> Self
            where
                Self: OperatorTrait,
            {
                // As for `__sub__`, this clone is load-bearing rather than wasteful: the in-place
                // operation appends to the buffers it is given.
                let mut result = self.clone();
                result.__iscaled_add__(other, factor);
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

    /// Asserts that clearing the group indices is reachable through the trait and always succeeds.
    ///
    /// The bound is the point of this helper: it can only resolve if `clear_groups` is reachable
    /// through the trait itself, which is what pins it as the single infallible route by which every
    /// operation that drops a grouping does so.
    fn assert_clear_groups_contract<T: OperatorTrait + OperatorMacro>() {
        let mut one = T::one();
        one.set_groups(Some(vec![0])).unwrap();
        assert!(one.has_groups());

        one.clear_groups();
        assert!(!one.has_groups());
        assert!(one.groups().is_none());

        // clearing an operator that tracks no group indices is a no-op rather than a panic
        one.clear_groups();
        assert!(!one.has_groups());
    }

    #[test]
    fn test_clear_groups_through_trait_bound() {
        assert_clear_groups_contract::<FermionOperator>();
        assert_clear_groups_contract::<MajoranaOperator>();
        assert_clear_groups_contract::<EdgeVertexOperator>();
        assert_clear_groups_contract::<TransferVertexOperator>();
    }

    /// Asserts that every in-place operation which changes the term count drops the grouping.
    ///
    /// This is the invariant the `groups` write discipline exists to protect:
    /// [`OperatorTrait::iter_with_groups`] zips the coefficients against the group indices, so an
    /// operation that changed the number of terms while leaving a stale array behind would silently
    /// drop the terms past its end.
    fn assert_term_count_changes_drop_groups<T: OperatorTrait + OperatorMacro>() {
        let mut op = T::one();
        op.set_groups(Some(vec![0])).unwrap();
        op.__iadd__(&T::one());
        assert!(!op.has_groups(), "__iadd__ must drop the grouping");

        let mut op = T::one();
        op.set_groups(Some(vec![0])).unwrap();
        op.__isub__(&T::one());
        assert!(!op.has_groups(), "__isub__ must drop the grouping");

        // scaled to zero first, so that `ichop` actually removes the term rather than no-op'ing
        let mut op = T::one();
        op.__imul__(Complex64::new(0.0, 0.0));
        op.set_groups(Some(vec![0])).unwrap();
        op.ichop(1e-8);
        assert!(!op.has_groups(), "ichop must drop the grouping");
    }

    #[test]
    fn test_term_count_changes_drop_groups() {
        assert_term_count_changes_drop_groups::<FermionOperator>();
        assert_term_count_changes_drop_groups::<MajoranaOperator>();
        assert_term_count_changes_drop_groups::<EdgeVertexOperator>();
        assert_term_count_changes_drop_groups::<TransferVertexOperator>();
    }
}
