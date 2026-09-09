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

use std::ops::Range;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GroupingError {
    #[error("the operator does not conform to an electronic structure observable")]
    ElectronicStructureError,
}

pub mod analysis;
pub mod electronic_structure;

/// Returns the half-open range of term indices carrying `group`, or `None` if no term does.
///
/// The result is meaningless unless `groups` is sorted: the two binary searches assume each group
/// occupies one contiguous run. Since this is called once per requested group, it does not re-check
/// that itself - a whole-array check here would scale with the terms held rather than the groups
/// requested, which is the very cost the search exists to avoid. Callers establish the precondition
/// with a single `groups.is_sorted()` and fall back to a linear scan otherwise.
///
/// Written once here rather than per operator type because only the *search* is shared; gathering the
/// terms in the returned range needs each type's own buffers.
pub(crate) fn group_term_range(groups: &[u32], group: u32) -> Option<Range<usize>> {
    let start = groups.partition_point(|&g| g < group);
    let end = groups.partition_point(|&g| g <= group);
    if start == end {
        // No term carries this group index. A sparse index range is legal (nothing enforces
        // denseness), so this is a normal outcome rather than an error.
        return None;
    }
    Some(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_term_range() {
        let groups = [0, 0, 1, 2, 2, 2];

        assert_eq!(group_term_range(&groups, 0), Some(0..2));
        assert_eq!(group_term_range(&groups, 1), Some(2..3));
        assert_eq!(group_term_range(&groups, 2), Some(3..6));
    }

    #[test]
    fn test_group_term_range_missing_index() {
        // A gap in the index range and an index past the end both mean "no term carries this".
        let groups = [0, 0, 2, 2];

        assert_eq!(group_term_range(&groups, 1), None);
        assert_eq!(group_term_range(&groups, 3), None);
    }

    #[test]
    fn test_group_term_range_empty() {
        assert_eq!(group_term_range(&[], 0), None);
    }
}
