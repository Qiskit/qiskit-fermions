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
use crate::operators::library::electronic_integrals::{From1Body, From2Body};
use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use regex::{Captures, Regex};
use std::fs::File;
use std::io::Read;

#[derive(Clone, Debug, PartialEq)]
pub struct FCIDump {
    pub norb: u32,
    pub nelec: u32,
    pub ms2: u32,
    pub constant: Option<f64>,
    pub one_body_a: Array1<f64>,
    pub one_body_b: Option<Array1<f64>>,
    pub two_body_aa: Array1<f64>,
    pub two_body_ab: Option<Array1<f64>>,
    pub two_body_bb: Option<Array1<f64>>,
}

impl FCIDump {
    pub fn from_file(file_path: String) -> Self {
        let mut file = match File::open(&file_path) {
            Err(why) => panic!("Could not open {}: {}", &file_path, why),
            Ok(file) => file,
        };
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let namelist_end = Regex::new(r"(/|&END)").unwrap();
        let Some(header) = namelist_end.captures(&contents) else {
            panic!("Could not find of HEADER in FCIDump file!")
        };
        let integrals = contents.split_off(header.get_match().end());

        let header_field = Regex::new(r"([^\s,]+)\s*\=\s*((?:[-+]?\d*\.\d+,|[-+]?\d+,)+)").unwrap();

        let mut _norb: Option<usize> = None;
        let mut _nelec: Option<usize> = None;
        let mut ms2: u32 = 0;
        // TODO: handle these remaining fields:
        // let mut isym: usize = 1;
        // let mut orbsym: Vec<usize> = vec![];
        // let mut iprtim: i64 = -1;
        // let mut int: usize = 5;
        // let mut memory: usize = 10_000;
        // let mut core: f64 = 0.0;
        // let mut maxit: usize = 25;
        // let mut thr: f64 = 1e-5;
        // let mut thrres: f64 = 0.1;
        // let mut nroot: usize = 1;

        fn unwrap_cap(cap: &Captures, idx: usize) -> String {
            cap.get(idx).unwrap().as_str().trim().replace(",", "")
        }

        for field in header_field.captures_iter(&contents) {
            match field.get(1).unwrap().as_str().to_lowercase().as_str() {
                "norb" => _norb = unwrap_cap(&field, 2).parse::<usize>().ok(),
                "nelec" => _nelec = unwrap_cap(&field, 2).parse::<usize>().ok(),
                "ms2" => ms2 = unwrap_cap(&field, 2).parse::<u32>().expect("Missing ms2!"),
                _ => continue,
            };
        }

        let norb = _norb.expect("Missing norb!");
        let nelec = _nelec.expect("Missing nelec!");
        let npair = norb * (norb + 1) / 2;
        let num_s4 = npair * npair;
        let num_s8 = npair * (npair + 1) / 2;

        let integral_line = Regex::new(r"(.+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)").unwrap();

        let mut beta_present: bool = false;
        let mut constant: Option<f64> = None;
        let mut one_body_a = Array1::<f64>::zeros(npair);
        let mut one_body_b = Array1::<f64>::zeros(npair);
        let mut two_body_aa = Array1::<f64>::zeros(num_s8);
        let mut two_body_ab = Array1::<f64>::zeros(num_s4);
        let mut two_body_bb = Array1::<f64>::zeros(num_s8);

        for line in integrals.lines() {
            let Some(integral) = integral_line.captures(line.trim()) else {
                continue;
            };
            let coeff = unwrap_cap(&integral, 1).parse::<f64>().unwrap();
            let i = unwrap_cap(&integral, 2).parse::<usize>().unwrap();
            let a = unwrap_cap(&integral, 3).parse::<usize>().unwrap();
            let j = unwrap_cap(&integral, 4).parse::<usize>().unwrap();
            let b = unwrap_cap(&integral, 5).parse::<usize>().unwrap();

            match (i, a, j, b) {
                (0, 0, 0, 0) => constant = Some(coeff),
                (_, _, 0, 0) => {
                    let (mut _i, mut _a) = (i - 1, a - 1);
                    if _i < _a {
                        (_i, _a) = (_a, _i);
                    }
                    if (_i, _a) < (norb, norb) {
                        let _ia = _i * (_i + 1) / 2 + _a;
                        one_body_a[_ia] = coeff;
                    } else {
                        beta_present = true;
                        let _ia = (_i - norb) * (_i - norb + 1) / 2 + (_a - norb);
                        one_body_b[_ia] = coeff;
                    }
                }
                (_, _, _, 0) => todo!("MO energy value"),
                (_, _, _, _) => {
                    let (mut _i, mut _a, mut _j, mut _b) = (i - 1, a - 1, j - 1, b - 1);
                    if _i < _a {
                        (_i, _a) = (_a, _i);
                    }
                    if _j < _b {
                        (_j, _b) = (_b, _j);
                    }
                    if (_j, _b) < (norb, norb) {
                        let mut _ia = _i * (_i + 1) / 2 + _a;
                        let mut _jb = _j * (_j + 1) / 2 + _b;
                        if _ia < _jb {
                            (_ia, _jb) = (_jb, _ia);
                        }
                        let _iajb = _ia * (_ia + 1) / 2 + _jb;
                        two_body_aa[_iajb] = coeff;
                    } else {
                        beta_present = true;
                        if (_i, _a) < (norb, norb) {
                            let mut _ia = _i * (_i + 1) / 2 + _a;
                            let mut _jb = (_j - norb) * (_j - norb + 1) / 2 + (_b - norb);
                            let _iajb = _ia * npair + _jb;
                            two_body_ab[_iajb] = coeff;
                        } else {
                            let mut _ia = (_i - norb) * (_i - norb + 1) / 2 + (_a - norb);
                            let mut _jb = (_j - norb) * (_j - norb + 1) / 2 + (_b - norb);
                            if _ia < _jb {
                                (_ia, _jb) = (_jb, _ia);
                            }
                            let _iajb = _ia * (_ia + 1) / 2 + _jb;
                            two_body_bb[_iajb] = coeff;
                        }
                    }
                }
            }
        }

        Self {
            norb: norb as u32,
            nelec: nelec as u32,
            ms2,
            constant,
            one_body_a,
            one_body_b: match beta_present {
                true => Some(one_body_b),
                false => None,
            },
            two_body_aa,
            two_body_ab: match beta_present {
                true => Some(two_body_ab),
                false => None,
            },
            two_body_bb: match beta_present {
                true => Some(two_body_bb),
                false => None,
            },
        }
    }
}

impl From<&FCIDump> for FermionOperator {
    fn from(fcidump: &FCIDump) -> Self {
        let mut op = Self::zero();

        if let Some(coeff) = fcidump.constant {
            op.coeffs.push(Complex64::new(coeff, 0.0));
            op.boundaries.push(op.boundaries.len() - 1);
        };

        match &fcidump.one_body_b {
            Some(_) => {
                op.add_1body_tril_spin(
                    ArrayView1::from(&fcidump.one_body_a),
                    ArrayView1::from(fcidump.one_body_b.as_ref().unwrap()),
                    fcidump.norb,
                );
                op.add_2body_tril_spin(
                    ArrayView1::from(&fcidump.two_body_aa),
                    ArrayView1::from(fcidump.two_body_ab.as_ref().unwrap()),
                    ArrayView1::from(fcidump.two_body_bb.as_ref().unwrap()),
                    fcidump.norb,
                );
            }
            None => {
                op.add_1body_tril_spin_sym(ArrayView1::from(&fcidump.one_body_a), fcidump.norb);
                op.add_2body_tril_spin_sym(ArrayView1::from(&fcidump.two_body_aa), fcidump.norb);
            }
        }

        op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_file() {
        let file_path = String::from("../../tests/h2.fcidump");
        let fcidump = FCIDump::from_file(file_path);

        let expected = FCIDump {
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

        assert_eq!(fcidump, expected);
    }

    #[test]
    fn test_to_fermion_operator() {
        let fcidump = FCIDump {
            norb: 2,
            nelec: 2,
            ms2: 0,
            constant: None,
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

        let expected = FermionOperator {
            coeffs: vec![
                -1.2563390730032502,
                -1.2563390730032502,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -2.3575299028703285e-16,
                -0.4718960072811406,
                -0.4718960072811406,
                0.33785507740175824,
                0.33785507740175824,
                0.33785507740175824,
                0.33785507740175824,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.09046559989211567,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.3322908651276483,
                0.34928686136600917,
                0.34928686136600917,
                0.34928686136600917,
                0.34928686136600917,
            ]
            .iter()
            .map(|c| Complex64::new(*c, 0.0))
            .collect(),
            actions: vec![
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false,
            ],
            indices: vec![
                0, 0, 2, 2, 1, 0, 0, 1, 3, 2, 2, 3, 1, 1, 3, 3, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0,
                2, 2, 2, 2, 1, 1, 0, 0, 3, 1, 0, 2, 1, 3, 2, 0, 3, 3, 2, 2, 0, 1, 0, 1, 2, 1, 0, 3,
                0, 3, 2, 1, 2, 3, 2, 3, 1, 0, 1, 0, 3, 0, 1, 2, 1, 2, 3, 0, 3, 2, 3, 2, 0, 0, 1, 1,
                2, 0, 1, 3, 0, 2, 3, 1, 2, 2, 3, 3, 1, 0, 0, 1, 3, 0, 0, 3, 1, 2, 2, 1, 3, 2, 2, 3,
                0, 1, 1, 0, 2, 1, 1, 2, 0, 3, 3, 0, 2, 3, 3, 2, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1,
                3, 3, 3, 3,
            ],
            boundaries: vec![
                0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68,
                72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140,
                144,
            ],
        };

        assert!(op.equiv(&expected, 1e-10));
    }

    #[test]
    fn test_from_file_beta() {
        let file_path = String::from("../../tests/heh.fcidump");
        let fcidump = FCIDump::from_file(file_path);

        let expected = FCIDump {
            norb: 2,
            nelec: 3,
            ms2: 1,
            constant: Some(1.4399379888979593),
            one_body_a: Array1::from_vec(vec![
                -2.6053045895340987,
                0.18301050723224974,
                -1.3466434111981145,
            ]),
            one_body_b: Some(Array1::from_vec(vec![
                -2.6172710340816154,
                0.13523295000711089,
                -1.334676966650596,
            ])),
            two_body_aa: Array1::from_vec(vec![
                0.9384381864717437,
                -0.17181645793946893,
                0.15284132948207071,
                0.6788914276314127,
                0.02923575406881855,
                0.7534734334131163,
            ]),
            two_body_ab: Some(Array1::from_vec(vec![
                0.950974621924185,
                -0.16158571043232664,
                0.6663549921789704,
                -0.1830105072322512,
                0.14486054289085748,
                0.040429803361600763,
                0.6768012137997059,
                0.026352760425216892,
                0.755563647244822,
            ])),
            two_body_bb: Some(Array1::from_vec(vec![
                0.9643310447658793,
                -0.17219894237602218,
                0.1373946928086696,
                0.6634447909580112,
                0.03696599236891227,
                0.7584738484657803,
            ])),
        };

        assert_eq!(fcidump, expected);
    }

    #[test]
    fn test_to_fermion_operator_beta() {
        let fcidump = FCIDump {
            norb: 2,
            nelec: 3,
            ms2: 1,
            constant: None,
            one_body_a: Array1::from_vec(vec![
                -2.6053045895340987,
                0.18301050723224974,
                -1.3466434111981145,
            ]),
            one_body_b: Some(Array1::from_vec(vec![
                -2.6172710340816154,
                0.13523295000711089,
                -1.334676966650596,
            ])),
            two_body_aa: Array1::from_vec(vec![
                0.9384381864717437,
                -0.17181645793946893,
                0.15284132948207071,
                0.6788914276314127,
                0.02923575406881855,
                0.7534734334131163,
            ]),
            two_body_ab: Some(Array1::from_vec(vec![
                0.950974621924185,
                -0.16158571043232664,
                0.6663549921789704,
                -0.1830105072322512,
                0.14486054289085748,
                0.040429803361600763,
                0.6768012137997059,
                0.026352760425216892,
                0.755563647244822,
            ])),
            two_body_bb: Some(Array1::from_vec(vec![
                0.9643310447658793,
                -0.17219894237602218,
                0.1373946928086696,
                0.6634447909580112,
                0.03696599236891227,
                0.7584738484657803,
            ])),
        };

        let op = FermionOperator::from(&fcidump);

        let expected = FermionOperator {
            coeffs: vec![
                -2.6053045895340987,
                0.18301050723224974,
                0.18301050723224974,
                -1.3466434111981145,
                -2.6172710340816154,
                0.13523295000711089,
                0.13523295000711089,
                -1.334676966650596,
                0.46921909323587185,
                -0.08590822896973446,
                -0.08590822896973446,
                -0.08590822896973446,
                -0.08590822896973446,
                0.07642066474103536,
                0.07642066474103536,
                0.07642066474103536,
                0.07642066474103536,
                0.3394457138157064,
                0.3394457138157064,
                0.014617877034409275,
                0.014617877034409275,
                0.014617877034409275,
                0.014617877034409275,
                0.3767367167065582,
                0.4754873109620925,
                0.4754873109620925,
                -0.08079285521616332,
                -0.08079285521616332,
                -0.08079285521616332,
                -0.08079285521616332,
                0.3331774960894852,
                0.3331774960894852,
                -0.0915052536161256,
                -0.0915052536161256,
                -0.0915052536161256,
                -0.0915052536161256,
                0.07243027144542874,
                0.07243027144542874,
                0.07243027144542874,
                0.07243027144542874,
                0.07243027144542874,
                0.07243027144542874,
                0.07243027144542874,
                0.07243027144542874,
                0.020214901680800382,
                0.020214901680800382,
                0.020214901680800382,
                0.020214901680800382,
                0.33840060689985296,
                0.33840060689985296,
                0.013176380212608446,
                0.013176380212608446,
                0.013176380212608446,
                0.013176380212608446,
                0.377781823622411,
                0.377781823622411,
                0.48216552238293964,
                -0.08609947118801109,
                -0.08609947118801109,
                -0.08609947118801109,
                -0.08609947118801109,
                0.0686973464043348,
                0.0686973464043348,
                0.0686973464043348,
                0.0686973464043348,
                0.3317223954790056,
                0.3317223954790056,
                0.018482996184456136,
                0.018482996184456136,
                0.018482996184456136,
                0.018482996184456136,
                0.3792369242328901,
            ]
            .iter()
            .map(|c| Complex64::new(*c, 0.0))
            .collect(),
            actions: vec![
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false, true, true, false, false, true, true,
                false, false, true, true, false, false,
            ],
            indices: vec![
                0, 0, 1, 0, 0, 1, 1, 1, 2, 2, 3, 2, 2, 3, 3, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1,
                0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0,
                2, 0, 0, 2, 0, 3, 2, 0, 3, 0, 0, 2, 0, 2, 3, 0, 2, 0, 0, 3, 0, 3, 3, 0, 3, 0, 0, 3,
                1, 2, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 2, 0, 1, 2, 1, 3, 2, 0, 3, 1, 0, 2, 0, 3, 2, 1,
                3, 0, 1, 2, 1, 2, 3, 0, 2, 1, 0, 3, 0, 2, 3, 1, 2, 0, 1, 3, 1, 3, 3, 0, 3, 1, 0, 3,
                0, 3, 3, 1, 3, 0, 1, 3, 1, 2, 2, 1, 2, 1, 1, 2, 1, 3, 2, 1, 3, 1, 1, 2, 1, 2, 3, 1,
                2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2,
                2, 2, 3, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 3, 2, 2, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2,
                3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3,
            ],
            boundaries: vec![
                0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68,
                72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140,
                144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                272,
            ],
        };

        println!("{op:#?}");
        assert!(op.equiv(&expected, 1e-10));
    }
}
