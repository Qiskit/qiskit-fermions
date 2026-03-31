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
            [true, true, false, false] => Ok(num_modes.pow(3) * min(term.modes[0], term.modes[3])
                + num_modes.pow(2) * min(term.modes[1], term.modes[2])
                + num_modes * max(term.modes[1], term.modes[2])
                + max(term.modes[0], term.modes[3])),
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
    use super::*;

    use crate::operators::library::fcidump::FCIDump;
    use ndarray::Array1;

    #[test]
    fn test_group_terms_by_electronic_structure() {
        let fcidump = FCIDump {
            norb: 2,
            nelec: 2,
            ms2: 0,
            constant: Some(0.7199689944489797),
            one_body_a: Array1::from_vec(vec![
                -1.2563390730032502,
                -2.3575299028703285E-16,
                -0.4718960072811406,
            ]),
            one_body_b: None,
            two_body_aa: Array1::from_vec(vec![
                0.6757101548035165,
                0.0,
                0.18093119978423133,
                0.6645817302552967,
                0.0,
                0.6985737227320183,
            ]),
            two_body_ab: None,
            two_body_bb: None,
        };

        let op = FermionOperator::from(&fcidump);

        let groups = op.split_out_groups();
        // no groups yet!
        assert!(groups.is_none());

        let mut normal = op.normal_ordered().simplify(1e-16);

        let _ = group_terms_by_electronic_structure(&mut normal, 2 * fcidump.norb);

        let groups = normal.split_out_groups().unwrap();
        assert!(groups.len() == 17);

        // TODO: We need to perform some actually useful assertion here! Ideally, we should use an
        // operator that is slightly larger to verify that 2-body terms get grouped correctly, too.
    }
}
