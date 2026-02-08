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

use num_complex::Complex64;

pub trait OperatorTrait {
    fn zero() -> Self;
    fn one() -> Self;
    fn equiv(&self, other: &Self, atol: f64) -> bool;

    fn adjoint(&self) -> Self;

    fn __iadd__(&mut self, other: &Self);
    fn __imul__(&mut self, other: Complex64);
    fn __iand__(&mut self, other: &Self);
    fn ichop(&mut self, atol: f64);
}

pub trait OperatorMacro {
    fn __add__(&self, other: &Self) -> Self;
    fn __sub__(&self, other: &Self) -> Self;
    fn __mul__(&self, other: Complex64) -> Self;
    fn __div__(&self, other: Complex64) -> Self;
    fn __neg__(&self) -> Self;
    fn __and__(&self, other: &Self) -> Self;
    fn __pow__(&self, exponent: usize) -> Self;

    // more in-place operations
    fn __isub__(&mut self, other: &Self);
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
                let mut result = self.clone();
                result.__iadd__(&other.__neg__());
                result
            }

            fn __isub__(&mut self, other: &Self)
            where
                Self: OperatorTrait,
            {
                self.__iadd__(&other.__neg__());
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
                let mut result = self.clone();
                result.__iand__(other);
                result
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

pub mod fermion_operator;
pub mod library;
pub mod majorana_operator;
