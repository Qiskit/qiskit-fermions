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
