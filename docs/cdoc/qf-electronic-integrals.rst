====================
Electronic Integrals
====================

.. meta::
   :description: C API constructor functions that build electronic structure Hamiltonians from
                 integral coefficients in several layouts (tril, full, spin, and sym).

The constructor functions listed here generally take the coefficients of
electronic structure Hamiltonians as an input. Different flavors exist:

* ``tril``: these functions consume 1-dimensional arrays of flattened
  (generalized) triangular indices
* ``full``: these functions consume high-dimensional arrays
* ``spin``: these functions take separate arrays for the different spin species
* ``sym``: these functions take a single array for one spin species and infer
  the other spin species

----

.. doxygengroup:: qf_electronic_integrals
   :content-only:
