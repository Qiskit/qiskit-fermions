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

use crate::operators::fermion_operator::{FermionOperator, FermionOperatorTermView};
use crate::operators::grouping::GroupingError;
use std::cmp::{max, min};
use std::collections::HashMap;

fn _hash_electronic_structure_term(
    term: FermionOperatorTermView,
    num_modes: u32,
) -> Result<u32, GroupingError> {
    match term.modes.len() {
        0 => Ok(num_modes.pow(4) + num_modes.pow(2)),
        2 => match term.actions {
            [true, false] => Ok(num_modes.pow(4)
                + num_modes * min(term.modes[0], term.modes[1])
                + max(term.modes[0], term.modes[1])),
            _ => Err(GroupingError::ElectronicStructureError),
        },
        4 => match term.actions {
            // TODO: expose configuration option for pairing in chemist vs. physicist ordering:
            //   - chemist:   i a j b
            //   - physicist: i a b j
            [true, true, false, false] => Ok(num_modes.pow(3) * min(term.modes[0], term.modes[2])
                + num_modes.pow(2) * min(term.modes[1], term.modes[3])
                + num_modes * max(term.modes[1], term.modes[3])
                + max(term.modes[0], term.modes[2])),
            _ => Err(GroupingError::ElectronicStructureError),
        },
        _ => Err(GroupingError::ElectronicStructureError),
    }
}

pub fn group_terms_by_electronic_structure(
    op: &mut FermionOperator,
    num_modes: u32,
) -> Result<(), GroupingError> {
    let mut groups = HashMap::new();
    let mut group_indices: Vec<u32> = Vec::with_capacity(op.coeffs.len());
    for term in op.iter() {
        let key = _hash_electronic_structure_term(term, num_modes)?;
        let num_groups = groups.len();
        let group_idx = groups.entry(key).or_insert(num_groups as u32);
        group_indices.push(*group_idx);
    }
    op.groups = Some(group_indices);
    Ok(())
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::*;

    use crate::operators::OperatorTrait;
    use crate::operators::library::fcidump::FCIDump;

    #[test]
    fn test_grouping_error() {
        let mut op = FermionOperator {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            actions: vec![true],
            modes: vec![0],
            boundaries: vec![0, 1],
            groups: None,
        };

        let err = group_terms_by_electronic_structure(&mut op, 2);
        assert!(err.is_err_and(|e| matches!(e, GroupingError::ElectronicStructureError)));
    }

    #[test]
    fn test_group_terms_by_electronic_structure() {
        let file_path = String::from("../../tests/h2.fcidump");
        let fcidump = FCIDump::from_file(file_path);

        let op = FermionOperator::from(&fcidump);

        let groups = op.split_out_groups();
        assert!(
            groups.is_none(),
            "We should not have any group indices yet!"
        );

        let mut normal = op.normal_ordered().simplify(1e-16);

        let res = group_terms_by_electronic_structure(&mut normal, 2 * fcidump.norb);
        assert!(res.is_ok(), "We should not have a GroupingError here!");
        assert!(normal.groups.is_some(), "Now we should have group indices!");
        assert!(
            *normal.groups.as_ref().unwrap().iter().max().unwrap() == 13,
            "The number of groups we expect is 14, meaning the highest group index should be 13!",
        );

        let groups = normal.split_out_groups().unwrap();
        assert!(
            groups.len() == 14,
            "Expected 14 individual operators, one for each group."
        );

        let mut expected = Vec::with_capacity(14);

        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(0.7199689944489797, 0.0)],
            actions: vec![],
            modes: vec![],
            boundaries: vec![0, 0],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-1.2563390730032502, 0.0)],
            actions: vec![true, false],
            modes: vec![0, 0],
            boundaries: vec![0, 2],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.4718960072811406, 0.0)],
            actions: vec![true, false],
            modes: vec![1, 1],
            boundaries: vec![0, 2],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-1.2563390730032502, 0.0)],
            actions: vec![true, false],
            modes: vec![2, 2],
            boundaries: vec![0, 2],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.4718960072811406, 0.0)],
            actions: vec![true, false],
            modes: vec![3, 3],
            boundaries: vec![0, 2],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![
                Complex64::new(-2.3575299028703285e-16, 0.0),
                Complex64::new(-2.3575299028703285e-16, 0.0),
            ],
            actions: vec![true, false, true, false],
            modes: vec![0, 1, 1, 0],
            boundaries: vec![0, 2, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![
                Complex64::new(-2.3575299028703285e-16, 0.0),
                Complex64::new(-2.3575299028703285e-16, 0.0),
            ],
            actions: vec![true, false, true, false],
            modes: vec![3, 2, 2, 3],
            boundaries: vec![0, 2, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.4836505304710653, 0.0)],
            actions: vec![true, true, false, false],
            modes: vec![1, 0, 1, 0],
            boundaries: vec![0, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.6757101548035165, 0.0)],
            actions: vec![true, true, false, false],
            modes: vec![2, 0, 2, 0],
            boundaries: vec![0, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.6645817302552967, 0.0)],
            actions: vec![true, true, false, false],
            modes: vec![3, 0, 3, 0],
            boundaries: vec![0, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.6645817302552967, 0.0)],
            actions: vec![true, true, false, false],
            modes: vec![2, 1, 2, 1],
            boundaries: vec![0, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.6985737227320183, 0.0)],
            actions: vec![true, true, false, false],
            modes: vec![3, 1, 3, 1],
            boundaries: vec![0, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![Complex64::new(-0.4836505304710653, 0.0)],
            actions: vec![true, true, false, false],
            modes: vec![3, 2, 3, 2],
            boundaries: vec![0, 4],
            groups: None,
        });
        expected.push(FermionOperator {
            coeffs: vec![
                Complex64::new(-0.18093119978423133, 0.0),
                Complex64::new(-0.18093119978423133, 0.0),
                Complex64::new(-0.18093119978423133, 0.0),
                Complex64::new(-0.18093119978423133, 0.0),
            ],
            actions: vec![
                true, true, false, false, true, true, false, false, true, true, false, false, true,
                true, false, false,
            ],
            modes: vec![2, 0, 3, 1, 2, 1, 3, 0, 3, 1, 2, 0, 3, 0, 2, 1],
            boundaries: vec![0, 4, 8, 12, 16],
            groups: None,
        });

        for group in groups.iter() {
            let prior_len = expected.len();
            // for each group, remove the equivalent operator from the vector of expected groups
            expected.retain(|e| !group.equiv(e, 1e-16));
            let new_len = expected.len();
            if new_len == prior_len {
                // if we do not remove a group this time, we did not find a matching operator, thus
                // we must fail the test.
                assert!(
                    false,
                    "Could not find a matching group operator in the expected set!"
                );
            }
        }
        // if the expected groups are not fully consumed, we also must fail!
        assert!(
            expected.len() == 0,
            "Did not generate a group operator for all expected groups!"
        );
    }
}
