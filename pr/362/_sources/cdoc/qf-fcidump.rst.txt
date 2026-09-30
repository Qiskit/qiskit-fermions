=======
FCIDump
=======

.. c:struct:: QfFCIDump

An electronic structure Hamiltonian in FCIDump format.

----

Definition
==========

The FCIDump format was originally defined by Knowles and Handy, 1989 [1]_.
It is a widespread format for exporting electronic structure Hamiltonians in a plain-text file.

The present data structure only stores the information relevant for constructing the second
quantized operator.
However, this implementation goes beyond the original definition by supporting unrestricted
spin data to be loaded. The table below outlines how integrals are associated with spin species
based on the intervals in which the indices fall (assuming a header with ``NORB=n``):

================== ================ ================ ================ ================
Integral Type      i                j                k                l
================== ================ ================ ================ ================
Constant           :math:`{0}`      :math:`{0}`      :math:`{0}`      :math:`{0}`
1-body alpha       :math:`{0}`      :math:`{0}`      :math:`[1,n]`    :math:`[1,n]`
1-body beta        :math:`{0}`      :math:`{0}`      :math:`[n+1,2n]` :math:`[n+1,2n]`
2-body alpha-alpha :math:`[1,n]`    :math:`[1,n]`    :math:`[1,n]`    :math:`[1,n]`
2-body alpha-beta  :math:`[1,n]`    :math:`[1,n]`    :math:`[n+1,2n]` :math:`[n+1,2n]`
2-body beta-beta   :math:`[n+1,2n]` :math:`[n+1,2n]` :math:`[n+1,2n]` :math:`[n+1,2n]`
================== ================ ================ ================ ================

The only required values are the 1-body alpha-spin integrals.

The two-body integrals are expected in `chemist` ordering, :math:`(ij|kl)`, matching the FCIDump
convention.

----

.. _qf_fcidump-implementation:

Data layout
===========

The integrals are stored as flattened arrays exploiting their permutational symmetry, which is also
the layout the electronic integral constructors (:doc:`qf-electronic-integrals`) expect. Writing
:math:`n` for the number of orbitals (:c:func:`qf_fcidump_norb`) and
:math:`\text{npair} = n (n + 1) / 2`, a single data structure contains up to 5 arrays:

=============== =========================================== ==========================================
Array           Length                                      Contents
=============== =========================================== ==========================================
``one_body_a``  :math:`\text{npair}`                        The :math:`\alpha`-spin 1-body integrals.
``one_body_b``  :math:`\text{npair}`                        The :math:`\beta`-spin 1-body integrals.
``two_body_aa`` :math:`\text{npair} (\text{npair} + 1) / 2` The :math:`\alpha\alpha` 2-body integrals.
``two_body_ab`` :math:`\text{npair}^2`                      The :math:`\alpha\beta` 2-body integrals.
``two_body_bb`` :math:`\text{npair} (\text{npair} + 1) / 2` The :math:`\beta\beta` 2-body integrals.
=============== =========================================== ==========================================

The 1-body arrays are the flattened lower triangle of an :math:`(n, n)` matrix, indexed as
``ia = i * (i + 1) / 2 + a`` with ``a <= i``. The ``two_body_aa`` and ``two_body_bb`` arrays are
8-fold (S8) symmetric: the flattened lower triangle of a
:math:`(\text{npair}, \text{npair})` matrix, whose own two axes are each such a pair index.

.. warning::
   ``two_body_ab`` is packed *differently* from its siblings. It is only 4-fold (S4) symmetric, so it
   holds the **full** :math:`(\text{npair}, \text{npair})` matrix in row-major order, indexed as
   ``iajb = ia * npair + jb``. Its row pair indexes the :math:`\alpha`-spin species and its column
   pair the :math:`\beta`-spin species, so it is *not* symmetric under exchanging the two pairs.

The :math:`\beta`-spin 1-body and the :math:`\alpha\beta` / :math:`\beta\beta` 2-body arrays are only
present for a file carrying unrestricted spin data. They are absent together, so
:c:func:`qf_fcidump_is_unrestricted` reports on all three at once and gates the getters that read
them.

The stored values are the integrals as they appear in the file. In particular, they do **not** carry
the conventional factor of :math:`\frac{1}{2}` that the 2-body operator terms do, so a coefficient of
an operator built via :c:func:`qf_ferm_op_from_fcidump` is half the corresponding value returned here.

Building a dense :math:`(n, n)` or :math:`(n, n, n, n)` tensor from these arrays (as required by, for
example, sample-based diagonalization tooling) is the caller's task; the index formulae above are what
that expansion needs.

----

.. [1] P. J. Knowles and N. C. Handy, Computer Physics Communications 54 (1989) 75-83.

Members
=======

.. doxygengroup:: qf_fcidump
   :content-only:
   :members:
   :undoc-members:

----

Conversion
==========

Operator representations which can be constructed from an instance of
:c:struct:`QfFCIDump` provide a ``qf_*_from_fcidump`` function:

.. doxygengroup:: qf_fcidump_constructors
   :content-only:
