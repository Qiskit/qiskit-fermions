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

use crate::operators::OperatorTrait;
use crate::operators::fermion_operator::FermionOperator;
use ndarray::ArrayView1;
use num_complex::Complex64;

fn _inflate_index(index: u32) -> (u32, u32) {
    let mut p = 0;
    while (p + 1) * (p + 2) / 2 <= index {
        p += 1;
    }
    let q = index - p * (p + 1) / 2;
    (p, q)
}

fn _expand_s4_index(iajb: u32, npair: u32) -> Vec<(u32, u32, u32, u32)> {
    let ia = iajb / npair;
    let jb = iajb % npair;

    let (i, a) = _inflate_index(ia);
    let (j, b) = _inflate_index(jb);

    let mut res = vec![(i, a, j, b)];
    if i > a {
        res.push((a, i, j, b));
    }
    if j > b {
        res.push((i, a, b, j));
    }
    if i > a && j > b {
        res.push((a, i, b, j));
    }
    res
}

fn _expand_s8_index(iajb: u32) -> Vec<(u32, u32, u32, u32)> {
    let (ia, jb) = _inflate_index(iajb);
    let (mut i, mut a) = _inflate_index(ia);
    let (mut j, mut b) = _inflate_index(jb);

    let mut res = vec![(i, a, j, b)];
    if i > a {
        res.push((a, i, j, b));
    }
    if j > b {
        res.push((i, a, b, j));
    }
    if i > a && j > b {
        res.push((a, i, b, j));
    }
    if ia > jb {
        (i, a, j, b) = (j, b, i, a);
        res.push((i, a, j, b));
        if i > a {
            res.push((a, i, j, b));
        }
        if j > b {
            res.push((i, a, b, j));
        }
        if i > a && j > b {
            res.push((a, i, b, j));
        }
    }
    res
}

pub trait From1Body {
    fn add_1body_tril_spin_sym(&mut self, one_body_a: ArrayView1<f64>, norb: u32);
    fn add_1body_tril_spin(
        &mut self,
        one_body_a: ArrayView1<f64>,
        one_body_b: ArrayView1<f64>,
        norb: u32,
    );

    fn from_1body_tril_spin_sym(one_body_a: ArrayView1<f64>, norb: u32) -> Self;
    fn from_1body_tril_spin(
        one_body_a: ArrayView1<f64>,
        one_body_b: ArrayView1<f64>,
        norb: u32,
    ) -> Self;

    // TODO:
    //  - from_1body_tril(one_body: ArrayView1<f64>, norb: u32) -> Self;
    //  - from_1body_full_spin_sym(one_body: ArrayView2<f64>) -> Self;
    //  - from_1body_full_spin(one_body_a: ArrayView2<f64>, one_body_b: ArrayView2<f64>) -> Self;
    //  - from_1body_full(one_body: ArrayView2<f64>) -> Self;
}

impl FermionOperator {
    #[inline]
    fn _insert_1body_idx(op: &mut Self, c: Complex64, i: u32, a: u32) {
        op.coeffs.push(c);
        op.actions.push(true);
        op.actions.push(false);
        op.indices.push(i);
        op.indices.push(a);
        op.boundaries.push(op.indices.len());
        if i != a {
            op.coeffs.push(c);
            op.actions.push(true);
            op.actions.push(false);
            op.indices.push(a);
            op.indices.push(i);
            op.boundaries.push(op.indices.len());
        }
    }

    #[inline]
    fn _insert_2body_idx(op: &mut Self, c: Complex64, i: u32, j: u32, b: u32, a: u32) {
        op.coeffs.push(c);
        op.actions.push(true);
        op.actions.push(true);
        op.actions.push(false);
        op.actions.push(false);
        op.indices.push(i);
        op.indices.push(j);
        op.indices.push(b);
        op.indices.push(a);
        op.boundaries.push(op.indices.len());
    }
}

impl From1Body for FermionOperator {
    fn add_1body_tril_spin_sym(&mut self, one_body_a: ArrayView1<f64>, norb: u32) {
        one_body_a
            .indexed_iter()
            .filter(|&(_, coeff)| coeff.abs() > 0.0)
            .for_each(|(ia, &coeff)| {
                let (i, a) = _inflate_index(ia as u32);
                let c = Complex64::new(coeff, 0.0);
                Self::_insert_1body_idx(self, c, i, a);
                Self::_insert_1body_idx(self, c, i + norb, a + norb);
            });
    }

    fn from_1body_tril_spin_sym(one_body_a: ArrayView1<f64>, norb: u32) -> Self {
        let mut op = Self::zero();
        op.add_1body_tril_spin_sym(one_body_a, norb);
        op
    }

    fn add_1body_tril_spin(
        &mut self,
        one_body_a: ArrayView1<f64>,
        one_body_b: ArrayView1<f64>,
        norb: u32,
    ) {
        one_body_a
            .indexed_iter()
            .filter(|&(_, coeff)| coeff.abs() > 0.0)
            .for_each(|(ia, &coeff)| {
                let (i, a) = _inflate_index(ia as u32);
                let c = Complex64::new(coeff, 0.0);
                Self::_insert_1body_idx(self, c, i, a);
            });

        one_body_b
            .indexed_iter()
            .filter(|&(_, coeff)| coeff.abs() > 0.0)
            .for_each(|(ia, &coeff)| {
                let (i, a) = _inflate_index(ia as u32);
                let c = Complex64::new(coeff, 0.0);
                Self::_insert_1body_idx(self, c, i + norb, a + norb);
            });
    }

    fn from_1body_tril_spin(
        one_body_a: ArrayView1<f64>,
        one_body_b: ArrayView1<f64>,
        norb: u32,
    ) -> Self {
        let mut op = Self::zero();
        op.add_1body_tril_spin(one_body_a, one_body_b, norb);
        op
    }
}

pub trait From2Body {
    fn add_2body_tril_spin_sym(&mut self, two_body_aa: ArrayView1<f64>, norb: u32);
    fn add_2body_tril_spin(
        &mut self,
        two_body_aa: ArrayView1<f64>,
        two_body_ab: ArrayView1<f64>,
        two_body_bb: ArrayView1<f64>,
        norb: u32,
    );

    fn from_2body_tril_spin_sym(two_body_aa: ArrayView1<f64>, norb: u32) -> Self;
    fn from_2body_tril_spin(
        two_body_aa: ArrayView1<f64>,
        two_body_ab: ArrayView1<f64>,
        two_body_bb: ArrayView1<f64>,
        norb: u32,
    ) -> Self;

    // TODO:
    //  - from_2body_tril(two_body: ArrayView1<f64>, norb: u32) -> Self;
    //  - from_2body_full_spin_sym(two_body: ArrayView4<f64>) -> Self;
    //  - from_2body_full_spin(two_body_aa: ArrayView4<f64>, two_body_ab: ArrayView4<f64>, two_body_bb: ArrayView4<f64>) -> Self;
    //  - from_2body_full(two_body: ArrayView4<f64>) -> Self;
}

impl From2Body for FermionOperator {
    fn add_2body_tril_spin_sym(&mut self, two_body_aa: ArrayView1<f64>, norb: u32) {
        two_body_aa
            .indexed_iter()
            .filter(|&(_, coeff)| coeff.abs() > 0.0)
            .for_each(|(iajb, &coeff)| {
                let c = Complex64::new(0.5 * coeff, 0.0);
                _expand_s8_index(iajb as u32)
                    .iter()
                    .for_each(|&(i, a, j, b)| {
                        Self::_insert_2body_idx(self, c, i, j, b, a);
                        Self::_insert_2body_idx(self, c, i + norb, j, b, a + norb);
                        Self::_insert_2body_idx(self, c, i, j + norb, b + norb, a);
                        Self::_insert_2body_idx(self, c, i + norb, j + norb, b + norb, a + norb);
                    });
            });
    }

    fn from_2body_tril_spin_sym(two_body_aa: ArrayView1<f64>, norb: u32) -> Self {
        let mut op = Self::zero();
        op.add_2body_tril_spin_sym(two_body_aa, norb);
        op
    }

    fn add_2body_tril_spin(
        &mut self,
        two_body_aa: ArrayView1<f64>,
        two_body_ab: ArrayView1<f64>,
        two_body_bb: ArrayView1<f64>,
        norb: u32,
    ) {
        two_body_aa
            .indexed_iter()
            .filter(|&(_, coeff)| coeff.abs() > 0.0)
            .for_each(|(iajb, &coeff)| {
                let c = Complex64::new(0.5 * coeff, 0.0);
                _expand_s8_index(iajb as u32)
                    .iter()
                    .for_each(|&(i, a, j, b)| {
                        Self::_insert_2body_idx(self, c, i, j, b, a);
                    });
            });

        let npair = norb * (norb + 1) / 2;
        two_body_ab
            .indexed_iter()
            .filter(|&(_, coeff)| coeff.abs() > 0.0)
            .for_each(|(iajb, &coeff)| {
                let c = Complex64::new(0.5 * coeff, 0.0);
                _expand_s4_index(iajb as u32, npair)
                    .iter()
                    .for_each(|&(i, a, j, b)| {
                        Self::_insert_2body_idx(self, c, i, j + norb, b + norb, a);
                        Self::_insert_2body_idx(self, c, j + norb, i, a, b + norb);
                    });
            });

        two_body_bb
            .indexed_iter()
            .filter(|&(_, coeff)| coeff.abs() > 0.0)
            .for_each(|(iajb, &coeff)| {
                let c = Complex64::new(0.5 * coeff, 0.0);
                _expand_s8_index(iajb as u32)
                    .iter()
                    .for_each(|&(i, a, j, b)| {
                        Self::_insert_2body_idx(self, c, i + norb, j + norb, b + norb, a + norb);
                    });
            });
    }

    fn from_2body_tril_spin(
        two_body_aa: ArrayView1<f64>,
        two_body_ab: ArrayView1<f64>,
        two_body_bb: ArrayView1<f64>,
        norb: u32,
    ) -> Self {
        let mut op = Self::zero();
        op.add_2body_tril_spin(two_body_aa, two_body_ab, two_body_bb, norb);
        op
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn test_1body_tril_spin_sym() {
        let norb = 2;
        let one_body_a = Array1::from_iter((1..4).map(|i| f64::from(i)));

        let op = FermionOperator::from_1body_tril_spin_sym(ArrayView1::from(&one_body_a), norb);

        let expected = FermionOperator {
            coeffs: vec![1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0]
                .iter()
                .map(|c| Complex64::new(*c, 0.0))
                .collect(),
            actions: vec![true, false].iter().cloned().cycle().take(16).collect(),
            indices: vec![0, 0, 2, 2, 1, 0, 0, 1, 3, 2, 2, 3, 1, 1, 3, 3],
            boundaries: vec![0, 2, 4, 6, 8, 10, 12, 14, 16],
        };

        assert_eq!(op, expected);
    }

    #[test]
    fn test_1body_tril_spin() {
        let norb = 2;
        let one_body_a = Array1::from_iter((1..4).map(|i| f64::from(i)));
        let one_body_b = Array1::from_iter((1..4).map(|i| f64::from(-i)));

        let op = FermionOperator::from_1body_tril_spin(
            ArrayView1::from(&one_body_a),
            ArrayView1::from(&one_body_b),
            norb,
        );

        let expected = FermionOperator {
            coeffs: vec![1.0, 2.0, 2.0, 3.0, -1.0, -2.0, -2.0, -3.0]
                .iter()
                .map(|c| Complex64::new(*c, 0.0))
                .collect(),
            actions: vec![true, false].iter().cloned().cycle().take(16).collect(),
            indices: vec![0, 0, 1, 0, 0, 1, 1, 1, 2, 2, 3, 2, 2, 3, 3, 3],
            boundaries: vec![0, 2, 4, 6, 8, 10, 12, 14, 16],
        };

        assert_eq!(op, expected);
    }

    #[test]
    fn test_2body_tril_spin_sym() {
        let norb = 2;
        let two_body_aa = Array1::from_iter((1..7).map(|i| f64::from(i)));

        let op = FermionOperator::from_2body_tril_spin_sym(ArrayView1::from(&two_body_aa), norb);

        let expected = FermionOperator {
            coeffs: vec![
                0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5,
                2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0,
            ]
            .iter()
            .map(|c| Complex64::new(*c, 0.0))
            .collect(),
            actions: vec![true, true, false, false]
                .iter()
                .cloned()
                .cycle()
                .take(256)
                .collect(),
            indices: vec![
                0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 1, 0, 0, 0, 3, 0, 0, 2, 1, 2, 2, 0,
                3, 2, 2, 2, 0, 0, 0, 1, 2, 0, 0, 3, 0, 2, 2, 1, 2, 2, 2, 3, 0, 1, 0, 0, 2, 1, 0, 2,
                0, 3, 2, 0, 2, 3, 2, 2, 0, 0, 1, 0, 2, 0, 1, 2, 0, 2, 3, 0, 2, 2, 3, 2, 1, 1, 0, 0,
                3, 1, 0, 2, 1, 3, 2, 0, 3, 3, 2, 2, 0, 1, 0, 1, 2, 1, 0, 3, 0, 3, 2, 1, 2, 3, 2, 3,
                1, 0, 1, 0, 3, 0, 1, 2, 1, 2, 3, 0, 3, 2, 3, 2, 0, 0, 1, 1, 2, 0, 1, 3, 0, 2, 3, 1,
                2, 2, 3, 3, 1, 0, 0, 1, 3, 0, 0, 3, 1, 2, 2, 1, 3, 2, 2, 3, 0, 1, 1, 0, 2, 1, 1, 2,
                0, 3, 3, 0, 2, 3, 3, 2, 1, 1, 0, 1, 3, 1, 0, 3, 1, 3, 2, 1, 3, 3, 2, 3, 1, 0, 1, 1,
                3, 0, 1, 3, 1, 2, 3, 1, 3, 2, 3, 3, 1, 1, 1, 0, 3, 1, 1, 2, 1, 3, 3, 0, 3, 3, 3, 2,
                0, 1, 1, 1, 2, 1, 1, 3, 0, 3, 3, 1, 2, 3, 3, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1,
                3, 3, 3, 3,
            ],
            boundaries: (0..257).step_by(4).collect(),
        };

        assert_eq!(op, expected);
    }

    #[test]
    fn test_2body_tril_spin() {
        let norb = 2;
        let two_body_aa = Array1::from_iter((1..7).map(|i| f64::from(i)));
        let two_body_ab = Array1::from_iter((11..20).map(|i| f64::from(i)));
        let two_body_bb = Array1::from_iter((1..7).map(|i| f64::from(-i)));

        let op = FermionOperator::from_2body_tril_spin(
            ArrayView1::from(&two_body_aa),
            ArrayView1::from(&two_body_ab),
            ArrayView1::from(&two_body_bb),
            norb,
        );

        let expected = FermionOperator {
            coeffs: vec![
                0.5, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 3.0,
                5.5, 5.5, 6.0, 6.0, 6.0, 6.0, 6.5, 6.5, 7.0, 7.0, 7.0, 7.0, 7.5, 7.5, 7.5, 7.5,
                7.5, 7.5, 7.5, 7.5, 8.0, 8.0, 8.0, 8.0, 8.5, 8.5, 9.0, 9.0, 9.0, 9.0, 9.5, 9.5,
                -0.5, -1.0, -1.0, -1.0, -1.0, -1.5, -1.5, -1.5, -1.5, -2.0, -2.0, -2.5, -2.5, -2.5,
                -2.5, -3.0,
            ]
            .iter()
            .map(|c| Complex64::new(*c, 0.0))
            .collect(),
            actions: vec![true, true, false, false]
                .iter()
                .cloned()
                .cycle()
                .take(256)
                .collect(),
            indices: vec![
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1,
                1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0, 2, 0, 0, 2, 0, 3, 2, 0, 3, 0, 0, 2, 0, 2, 3, 0,
                2, 0, 0, 3, 0, 3, 3, 0, 3, 0, 0, 3, 1, 2, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 2, 0, 1, 2,
                1, 3, 2, 0, 3, 1, 0, 2, 0, 3, 2, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 1, 0, 3, 0, 2, 3, 1,
                2, 0, 1, 3, 1, 3, 3, 0, 3, 1, 0, 3, 0, 3, 3, 1, 3, 0, 1, 3, 1, 2, 2, 1, 2, 1, 1, 2,
                1, 3, 2, 1, 3, 1, 1, 2, 1, 2, 3, 1, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 2, 2, 2, 2,
                3, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 3, 2,
                2, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3,
                3, 3, 3, 3,
            ],
            boundaries: (0..257).step_by(4).collect(),
        };

        assert_eq!(op, expected);
    }
}
