<a id="module-qiskit_fermions.mappers.library"></a>

# Mapper Library

This module provides efficient implementations of commonly used operator representation mapper
routines.

## Python wrappers

The functions below are thin, type-agnostic convenience mappers. Each delegates to the appropriate
library implementation below under the hood, based on the type of the object it is given. They do
not implement any conversion logic themselves.

## Library implementations

The functions below are the native, efficient mapper implementations backing this module.
